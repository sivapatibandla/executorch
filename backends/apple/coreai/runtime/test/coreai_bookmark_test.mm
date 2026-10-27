/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <chrono>
#include <future>
#include <thread>
#include "coreai_bookmark_fixture.h"
#include "coreai_fault_scope.h"
#include "coreai_filesystem_fixture.h"
#include "coreai_manifest_fixture.h"
#include "coreai_storage.h"

using namespace executorch::runtime;
using namespace executorch::backends::coreai;
using namespace executorch::backends::coreai::testing;

namespace {
constexpr size_t kMiB = 1024 * 1024;
constexpr auto kWaitTimeout = std::chrono::seconds(5);

struct BookmarkWaiter {
  std::future<void> attempting;
  std::future<Result<std::unique_ptr<BookmarkLock>>> result;
};

BookmarkWaiter
start_bookmark_waiter(NSString* path, NSString* key, bool create = true) {
  std::promise<void> attempting;
  std::promise<Result<std::unique_ptr<BookmarkLock>>> result;
  BookmarkWaiter waiter{attempting.get_future(), result.get_future()};
  // Promise-backed futures cannot block teardown if acquisition stalls.
  std::thread([path,
               key,
               create,
               attempting = std::move(attempting),
               result = std::move(result)]() mutable {
    @autoreleasepool {
      StorageFaultScope scope(StorageOperation::BeforeBookmarkLock);
      scope.armed = false;
      auto* signal = &attempting;
      scope.observe = ^(StorageOperation operation) {
        if (operation == StorageOperation::BeforeBookmarkLock) {
          signal->set_value();
        }
      };
      result.set_value(lock_bookmark(path, key, create));
    }
  }).detach();
  return waiter;
}

class CoreAIBookmarkIdentityTest : public ::testing::Test {
 protected:
  Manifest manifest;
  NSString* key = nil;

  void SetUp() override {
    auto selected = bookmark_manifest();
    ASSERT_TRUE(selected.ok());
    manifest = selected.get();
    auto result = test_bookmark_key(manifest);
    ASSERT_TRUE(result.ok());
    key = result.get();
  }
};

class CoreAIBookmarkPathTest : public CoreAIBookmarkIdentityTest {
 protected:
  TestDirectory root;
  void SetUp() override {
    CoreAIBookmarkIdentityTest::SetUp();
    if (HasFatalFailure())
      return;
    ASSERT_NE(root.url, nil);
  }
  NSURL* lock_url() {
    return [[root.url URLByAppendingPathComponent:@"locks"]
        URLByAppendingPathComponent:[key stringByAppendingString:@".lock"]];
  }
};

class CoreAIBookmarkRawTest : public CoreAIBookmarkPathTest {
 protected:
  std::unique_ptr<BookmarkLock> locked;
  NSURL* url = nil;
  void SetUp() override {
    CoreAIBookmarkPathTest::SetUp();
    if (HasFatalFailure())
      return;
    auto result = lock_bookmark(root.url.path, key);
    ASSERT_TRUE(result.ok());
    locked = std::move(result.get());
    ASSERT_NE(locked, nullptr);
    url = test_bookmark_url(root.url, key);
  }
};

::testing::AssertionResult bookmark_bytes_equal(
    const BookmarkLock& lock,
    NSData* expected) {
  auto result = read_bookmark(lock);
  if (!result.ok())
    return ::testing::AssertionFailure()
        << "Bookmark read error " << int(result.error());
  if (expected == nil ? result.get() != nil
                      : ![result.get() isEqual:expected]) {
    return ::testing::AssertionFailure() << "Bookmark bytes differ";
  }
  return ::testing::AssertionSuccess();
}

::testing::AssertionResult no_temporary_bookmarks(NSURL* root) {
  NSURL* directory = [root URLByAppendingPathComponent:@"bookmarks"];
  FileDescriptor fd(open(
      directory.fileSystemRepresentation, O_RDONLY | O_DIRECTORY | O_CLOEXEC));
  if (fd.get() < 0)
    return ::testing::AssertionFailure()
        << "open bookmark directory: " << errno;
  auto children = storage_children(fd.get());
  if (!children.ok())
    return ::testing::AssertionFailure()
        << "Cannot enumerate bookmark directory";
  for (NSString* name in children.get()) {
    if ([name hasPrefix:@".tmp-"])
      return ::testing::AssertionFailure() << "Leaked " << name.UTF8String;
  }
  return ::testing::AssertionSuccess();
}

NSURL* identity_url(NSURL* root, NSString* key) {
  return [[root URLByAppendingPathComponent:@"bookmarks"]
      URLByAppendingPathComponent:[key stringByAppendingString:@".metadata"]];
}

TEST_F(CoreAIBookmarkIdentityTest, IgnoresFunctionBindingsAndDeliveryLocation) {
  EXPECT_EQ(key.length, 64u);
  manifest.function = @"another_function";
  manifest.inputs = @[ @"different_input" ];
  manifest.outputs = @[ @"different_output" ];
  manifest.hash = @"another_delivery_location";
  auto changed = test_bookmark_key(manifest);
  ASSERT_TRUE(changed.ok());
  EXPECT_TRUE([changed.get() isEqual:key]);
}

TEST_F(
    CoreAIBookmarkIdentityTest,
    KeySeparatesPlatformArchitectureWeightsAndAotContext) {
  auto platform = bookmark_key(manifest, @"iOS", @"arch_b");
  auto architecture = bookmark_key(manifest, @"macOS", @"arch_a");
  Manifest reweighted = manifest;
  reweighted.bundle_digests = @{@"model.aimodel" : fixture_identity('2')};
  auto weights = test_bookmark_key(reweighted);
  auto aot_manifest = bookmark_manifest(true);
  ASSERT_TRUE(aot_manifest.ok());
  auto aot = test_bookmark_key(aot_manifest.get());
  for (auto* changed : {&platform, &architecture, &weights, &aot}) {
    ASSERT_TRUE(changed->ok());
    EXPECT_FALSE([changed->get() isEqual:key]);
  }
}

TEST_F(CoreAIBookmarkIdentityTest, RejectsMissingArchitecture) {
  EXPECT_FALSE(bookmark_key(manifest, @"macOS", nil).ok());
  EXPECT_FALSE(bookmark_key(manifest, @"macOS", @"").ok());
}

TEST_F(CoreAIBookmarkIdentityTest, RejectsUnselectedBundle) {
  manifest.path = @"../escape";
  EXPECT_FALSE(test_bookmark_key(manifest).ok());
}

TEST_F(CoreAIBookmarkRawTest, MissingBookmarkIsAnEmptySuccessfulRead) {
  EXPECT_TRUE(bookmark_bytes_equal(*locked, nil));
  EXPECT_FALSE([NSFileManager.defaultManager fileExistsAtPath:url.path]);
}

TEST_F(CoreAIBookmarkRawTest, RoundTripsMaximumBookmarkAndReopens) {
  auto bytes = sized_bookmark(8 * kMiB);
  ASSERT_TRUE(bytes.ok());
  ASSERT_EQ(write_bookmark(*locked, bytes.get()), Error::Ok);
  EXPECT_TRUE(bookmark_bytes_equal(*locked, bytes.get()));
  EXPECT_TRUE([[NSData dataWithContentsOfURL:url] isEqual:bytes.get()]);
  locked.reset();
  auto saved = saved_bookmark(root.url, key);
  ASSERT_TRUE(saved.ok());
  EXPECT_TRUE([saved.get() isEqual:bytes.get()]);
}

TEST_F(CoreAIBookmarkRawTest, InvalidWritesPreserveMaximumBookmark) {
  auto maximum = sized_bookmark(8 * kMiB);
  ASSERT_TRUE(maximum.ok());
  ASSERT_EQ(write_bookmark(*locked, maximum.get()), Error::Ok);
  auto oversized = sized_bookmark(8 * kMiB + 1);
  ASSERT_TRUE(oversized.ok());
  for (NSData* invalid : {[NSData data], oversized.get()}) {
    SCOPED_TRACE(invalid.length);
    EXPECT_EQ(write_bookmark(*locked, invalid), Error::InvalidExternalData);
    EXPECT_TRUE([[NSData dataWithContentsOfURL:url] isEqual:maximum.get()]);
    EXPECT_TRUE(bookmark_bytes_equal(*locked, maximum.get()));
    EXPECT_TRUE(no_temporary_bookmarks(root.url));
  }
}

TEST_F(CoreAIBookmarkRawTest, RejectsEmptyStoredBookmark) {
  FileDescriptor file(open(
      url.fileSystemRepresentation,
      O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
      0600));
  ASSERT_GE(file.get(), 0);
  EXPECT_EQ(read_bookmark(*locked).error(), Error::InvalidExternalData);
}

TEST_F(CoreAIBookmarkRawTest, RejectsSymlinkedBookmarkWithoutReadingTarget) {
  NSURL* target = [root.url URLByAppendingPathComponent:@"target"];
  NSData* bytes = [@"target bytes" dataUsingEncoding:NSUTF8StringEncoding];
  ASSERT_TRUE([bytes writeToURL:target atomically:NO]);
  ASSERT_EQ(
      symlink(target.fileSystemRepresentation, url.fileSystemRepresentation),
      0);
  EXPECT_EQ(read_bookmark(*locked).error(), Error::AccessFailed);
  EXPECT_TRUE(([[NSData dataWithContentsOfURL:target] isEqual:bytes]));
}

TEST_F(CoreAIBookmarkRawTest, RejectsDirectoryAtBookmarkPath) {
  ASSERT_EQ(mkdir(url.fileSystemRepresentation, 0700), 0);
  EXPECT_EQ(read_bookmark(*locked).error(), Error::InvalidExternalData);
}

TEST_F(CoreAIBookmarkRawTest, RejectsFifoWithoutWaitingForWriter) {
  ASSERT_EQ(mkfifo(url.fileSystemRepresentation, 0600), 0);
  EXPECT_EQ(read_bookmark(*locked).error(), Error::InvalidExternalData);
}

TEST_F(CoreAIBookmarkRawTest, RejectsInaccessibleBookmark) {
  if (geteuid() == 0) {
    GTEST_SKIP() << "Root can bypass file read permissions";
  }
  NSData* bytes = [@"bookmark" dataUsingEncoding:NSUTF8StringEncoding];
  ASSERT_EQ(write_bookmark(*locked, bytes), Error::Ok);
  ASSERT_EQ(chmod(url.fileSystemRepresentation, 0000), 0);
  EXPECT_EQ(read_bookmark(*locked).error(), Error::AccessFailed);
  ASSERT_EQ(chmod(url.fileSystemRepresentation, 0600), 0);
  EXPECT_TRUE(bookmark_bytes_equal(*locked, bytes));
}

TEST_F(CoreAIBookmarkRawTest, NoncreatingLookupReadsExistingBookmark) {
  NSData* bytes = [@"bookmark" dataUsingEncoding:NSUTF8StringEncoding];
  ASSERT_EQ(write_bookmark(*locked, bytes), Error::Ok);
  locked.reset();
  auto existing = lock_bookmark(root.url.path, key, false);
  ASSERT_TRUE(existing.ok());
  ASSERT_NE(existing.get(), nullptr);
  EXPECT_TRUE(bookmark_bytes_equal(*existing.get(), bytes));
  EXPECT_FALSE([NSFileManager.defaultManager
      fileExistsAtPath:[root.url URLByAppendingPathComponent:@"staging"].path]);
}

TEST_F(CoreAIBookmarkRawTest, PreparesSharedStagingParentOnDemand) {
  NSString* expected = [root.url URLByAppendingPathComponent:@"staging"].path;
  EXPECT_FALSE([NSFileManager.defaultManager fileExistsAtPath:expected]);
  auto staging = prepare_bookmark_staging(*locked);
  ASSERT_TRUE(staging.ok());
  EXPECT_TRUE([staging.get() isEqual:expected]);
  struct stat info {};
  ASSERT_EQ(lstat(staging.get().fileSystemRepresentation, &info), 0);
  EXPECT_TRUE(S_ISDIR(info.st_mode));
  auto repeated = prepare_bookmark_staging(*locked);
  ASSERT_TRUE(repeated.ok());
  EXPECT_TRUE([repeated.get() isEqual:expected]);
  EXPECT_FALSE([NSFileManager.defaultManager
      fileExistsAtPath:[expected stringByAppendingPathComponent:key]]);
}

TEST_F(CoreAIBookmarkRawTest, RejectsSymlinkedStagingParent) {
  TestDirectory target;
  ASSERT_NE(target.url, nil);
  NSURL* staging = [root.url URLByAppendingPathComponent:@"staging"];
  ASSERT_EQ(
      symlink(
          target.url.fileSystemRepresentation,
          staging.fileSystemRepresentation),
      0);
  EXPECT_EQ(prepare_bookmark_staging(*locked).error(), Error::AccessFailed);
  NSArray* contents =
      [NSFileManager.defaultManager contentsOfDirectoryAtPath:target.url.path
                                                        error:nil];
  ASSERT_NE(contents, nil);
  EXPECT_EQ(contents.count, 0u);
}

TEST_F(CoreAIBookmarkRawTest, RejectsFileAtStagingParent) {
  NSURL* staging = [root.url URLByAppendingPathComponent:@"staging"];
  NSData* bytes = [@"not a directory" dataUsingEncoding:NSUTF8StringEncoding];
  ASSERT_TRUE([bytes writeToURL:staging atomically:NO]);
  EXPECT_EQ(prepare_bookmark_staging(*locked).error(), Error::AccessFailed);
  EXPECT_TRUE(([[NSData dataWithContentsOfURL:staging] isEqual:bytes]));
}

TEST_F(CoreAIBookmarkRawTest, RejectsOversizedStoredBookmark) {
  FileDescriptor file(open(
      url.fileSystemRepresentation,
      O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
      0600));
  ASSERT_GE(file.get(), 0);
  ASSERT_EQ(ftruncate(file.get(), 8 * kMiB + 1), 0);
  ASSERT_EQ(close(file.release()), 0);
  EXPECT_EQ(read_bookmark(*locked).error(), Error::InvalidExternalData);
  struct stat info {};
  ASSERT_EQ(lstat(url.fileSystemRepresentation, &info), 0);
  EXPECT_EQ(info.st_size, static_cast<off_t>(8 * kMiB + 1));
}

TEST_F(CoreAIBookmarkPathTest, AllowsTemporaryRootsAndRejectsInvalidKeys) {
  auto allowed = lock_bookmark(root.url.path, key);
  ASSERT_TRUE(allowed.ok());
  ASSERT_NE(allowed.get(), nullptr);
  EXPECT_FALSE(lock_bookmark(root.url.path, @"../escape").ok());
}

TEST_F(CoreAIBookmarkPathTest, ExcludesRootWithoutCreatingStaging) {
  auto locked = lock_bookmark(root.url.path, key);
  ASSERT_TRUE(locked.ok());
  EXPECT_TRUE(backup_excluded(root.url));
  EXPECT_FALSE([NSFileManager.defaultManager
      fileExistsAtPath:[root.url URLByAppendingPathComponent:@"staging"].path]);
}

TEST_F(
    CoreAIBookmarkPathTest,
    NoncreatingLookupLeavesMissingRootAndEntryUntouched) {
  ASSERT_TRUE(set_backup_excluded(root.url, false));
  NSString* missing = [root.url URLByAppendingPathComponent:@"missing"].path;
  for (NSString* path in @[ root.url.path, missing ]) {
    SCOPED_TRACE(path.UTF8String);
    auto resolved = resolve_bookmark_root(path, false);
    ASSERT_TRUE(resolved.ok());
    EXPECT_TRUE([resolved.get() isEqual:path]);
    auto absent = lock_bookmark(path, key, false);
    ASSERT_TRUE(absent.ok());
    EXPECT_EQ(absent.get(), nullptr);
  }
  NSArray* contents =
      [NSFileManager.defaultManager contentsOfDirectoryAtPath:root.url.path
                                                        error:nil];
  ASSERT_NE(contents, nil);
  EXPECT_EQ(contents.count, 0u);
  EXPECT_FALSE(backup_excluded(root.url));
}

TEST_F(CoreAIBookmarkPathTest, NoncreatingLookupIgnoresOtherEntries) {
  NSString* other_key = fixture_identity('f');
  auto other = lock_bookmark(root.url.path, other_key);
  ASSERT_TRUE(other.ok());
  ASSERT_NE(other.get(), nullptr);
  NSData* bytes = [@"other" dataUsingEncoding:NSUTF8StringEncoding];
  ASSERT_EQ(write_bookmark(*other.get(), bytes), Error::Ok);
  auto absent = lock_bookmark(root.url.path, key, false);
  ASSERT_TRUE(absent.ok());
  EXPECT_EQ(absent.get(), nullptr);
  EXPECT_FALSE([NSFileManager.defaultManager fileExistsAtPath:lock_url().path]);
  EXPECT_TRUE(bookmark_bytes_equal(*other.get(), bytes));
}

TEST_F(CoreAIBookmarkPathTest, NoncreatingLookupLocksStagingOnlyEntry) {
  NSURL* staged = [[root.url URLByAppendingPathComponent:@"staging"]
      URLByAppendingPathComponent:key];
  ASSERT_TRUE([NSFileManager.defaultManager createDirectoryAtURL:staged
                                     withIntermediateDirectories:YES
                                                      attributes:nil
                                                           error:nil]);
  auto existing = lock_bookmark(root.url.path, key, false);
  ASSERT_TRUE(existing.ok());
  ASSERT_NE(existing.get(), nullptr);
  EXPECT_TRUE(bookmark_bytes_equal(*existing.get(), nil));
  NSData* bytes = [@"bookmark" dataUsingEncoding:NSUTF8StringEncoding];
  EXPECT_EQ(write_bookmark(*existing.get(), bytes), Error::InvalidArgument);
  EXPECT_TRUE([NSFileManager.defaultManager fileExistsAtPath:lock_url().path]);
  EXPECT_FALSE([NSFileManager.defaultManager
      fileExistsAtPath:[root.url URLByAppendingPathComponent:@"bookmarks"]
                           .path]);
}

TEST_F(CoreAIBookmarkPathTest, ResolvesAndCreatesRootWithBackupExclusion) {
  ASSERT_TRUE(set_backup_excluded(root.url, false));
  NSString* path = [root.url URLByAppendingPathComponent:@"nested/assets"].path;
  auto resolved = resolve_bookmark_root(path);
  ASSERT_TRUE(resolved.ok());
  EXPECT_TRUE([resolved.get() isEqual:path]);
  EXPECT_TRUE(backup_excluded([NSURL fileURLWithPath:resolved.get()]));
  EXPECT_FALSE(backup_excluded(root.url));
  auto existing = resolve_bookmark_root(path, false);
  ASSERT_TRUE(existing.ok());
  EXPECT_TRUE([existing.get() isEqual:path]);
}

TEST_F(CoreAIBookmarkPathTest, RejectsInvalidResolvedRoots) {
  for (bool create : {false, true}) {
    for (NSString* path in
         @[ @"relative", @"~", @"~/cache", @"~nobody/cache" ]) {
      SCOPED_TRACE(path.UTF8String);
      EXPECT_EQ(
          resolve_bookmark_root(path, create).error(), Error::InvalidArgument);
    }
    EXPECT_EQ(
        resolve_bookmark_root(nil, create).error(), Error::InvalidArgument);
  }
  NSURL* file = [root.url URLByAppendingPathComponent:@"file"];
  ASSERT_TRUE([[NSData data] writeToURL:file atomically:NO]);
  EXPECT_EQ(
      resolve_bookmark_root(file.path, false).error(), Error::AccessFailed);
  EXPECT_EQ(
      resolve_bookmark_root(
          [file.path stringByAppendingPathComponent:@"child"], false)
          .error(),
      Error::AccessFailed);
}

TEST_F(
    CoreAIBookmarkPathTest,
    ThreadContentionSerializesSameKeyButNotDifferentKeys) {
  auto holder = lock_bookmark(root.url.path, key);
  ASSERT_TRUE(holder.ok());
  ASSERT_NE(holder.get(), nullptr);
  auto waiter = start_bookmark_waiter(root.url.path, key);
  ASSERT_EQ(
      waiter.attempting.wait_for(kWaitTimeout), std::future_status::ready);
  EXPECT_EQ(
      waiter.result.wait_for(std::chrono::milliseconds(100)),
      std::future_status::timeout);
  auto independent =
      start_bookmark_waiter(root.url.path, fixture_identity('f'));
  ASSERT_EQ(
      independent.result.wait_for(kWaitTimeout), std::future_status::ready);
  auto other = independent.result.get();
  ASSERT_TRUE(other.ok());
  ASSERT_NE(other.get(), nullptr);
  holder.get().reset();
  ASSERT_EQ(waiter.result.wait_for(kWaitTimeout), std::future_status::ready);
  auto acquired = waiter.result.get();
  ASSERT_TRUE(acquired.ok());
  EXPECT_NE(acquired.get(), nullptr);
}

TEST_F(CoreAIBookmarkRawTest, NoncreatingWaiterRechecksEvictedEntry) {
  NSData* bytes = [@"bookmark" dataUsingEncoding:NSUTF8StringEncoding];
  ASSERT_EQ(write_bookmark(*locked, bytes), Error::Ok);
  auto waiter = start_bookmark_waiter(root.url.path, key, false);
  ASSERT_EQ(
      waiter.attempting.wait_for(kWaitTimeout), std::future_status::ready);
  ASSERT_EQ(
      waiter.result.wait_for(std::chrono::milliseconds(100)),
      std::future_status::timeout);
  ASSERT_EQ(unlink(url.fileSystemRepresentation), 0);
  locked.reset();
  ASSERT_EQ(waiter.result.wait_for(kWaitTimeout), std::future_status::ready);
  auto absent = waiter.result.get();
  ASSERT_TRUE(absent.ok());
  EXPECT_EQ(absent.get(), nullptr);
}

TEST_F(CoreAIBookmarkRawTest, WaiterOpensBookmarkDirectoryAfterAcquiringLock) {
  auto waiter = start_bookmark_waiter(root.url.path, key);
  ASSERT_EQ(
      waiter.attempting.wait_for(kWaitTimeout), std::future_status::ready);
  ASSERT_EQ(
      waiter.result.wait_for(std::chrono::milliseconds(100)),
      std::future_status::timeout);
  NSURL* bookmarks = [root.url URLByAppendingPathComponent:@"bookmarks"];
  ASSERT_EQ(rmdir(bookmarks.fileSystemRepresentation), 0);
  ASSERT_EQ(mkdir(bookmarks.fileSystemRepresentation, 0700), 0);
  locked.reset();
  ASSERT_EQ(waiter.result.wait_for(kWaitTimeout), std::future_status::ready);
  auto acquired = waiter.result.get();
  ASSERT_TRUE(acquired.ok());
  ASSERT_NE(acquired.get(), nullptr);
  NSData* bytes = [@"bookmark" dataUsingEncoding:NSUTF8StringEncoding];
  ASSERT_EQ(write_bookmark(*acquired.get(), bytes), Error::Ok);
  EXPECT_TRUE([[NSData dataWithContentsOfURL:url] isEqual:bytes]);
}

TEST_F(
    CoreAIBookmarkPathTest,
    ProcessContentionAllowsIndependentKeyAndRecoversAfterHolderDeath) {
  BookmarkChild holder;
  ASSERT_TRUE(holder.spawn(@"--bookmark-hold", root.url.path, @"same"));
  ASSERT_TRUE(holder.receive('L'));
  NSURL* file = lock_url();
  struct stat before {};
  ASSERT_EQ(stat(file.fileSystemRepresentation, &before), 0);
  BookmarkChild waiter;
  ASSERT_TRUE(waiter.spawn(@"--bookmark-lock", root.url.path, @"same"));
  ASSERT_TRUE(waiter.receive('B'));
  ASSERT_TRUE(waiter.expect_blocked());
  BookmarkChild independent;
  ASSERT_TRUE(independent.spawn(@"--bookmark-lock", root.url.path, @"other"));
  ASSERT_TRUE(independent.receive('L'));
  ASSERT_TRUE(independent.expect_exit(0));
  ASSERT_TRUE(holder.kill_and_reap());
  ASSERT_TRUE(waiter.receive('L'));
  ASSERT_TRUE(waiter.expect_exit(0));
  struct stat after {};
  ASSERT_EQ(stat(file.fileSystemRepresentation, &after), 0);
  EXPECT_EQ(before.st_dev, after.st_dev);
  EXPECT_EQ(before.st_ino, after.st_ino);
}

TEST_F(CoreAIBookmarkIdentityTest, IdentityExcludesContextAndFunction) {
  NSString* bundle = manifest.path.lastPathComponent;
  auto identity = bookmark_identity(manifest, bundle);
  ASSERT_TRUE(identity.ok());
  EXPECT_EQ(identity.get().length, 64u);
  manifest.function = @"another_function";
  manifest.inputs = @[ @"different_input" ];
  manifest.outputs = @[ @"different_output" ];
  manifest.hash = @"another_delivery_location";
  manifest.minimum_version = (NSOperatingSystemVersion){27, 0, 0};
  auto unchanged = bookmark_identity(manifest, bundle);
  ASSERT_TRUE(unchanged.ok());
  EXPECT_TRUE([unchanged.get() isEqual:identity.get()]);
  auto context = bookmark_key(manifest, @"iOS", @"arch_a");
  ASSERT_TRUE(context.ok());
  EXPECT_FALSE([context.get() isEqual:key]);
}

TEST_F(CoreAIBookmarkIdentityTest, IdentitySeparatesDigestBundleAndKind) {
  NSString* bundle = manifest.path.lastPathComponent;
  auto identity = bookmark_identity(manifest, bundle);
  ASSERT_TRUE(identity.ok());
  Manifest reweighted = manifest;
  reweighted.bundle_digests = @{bundle : fixture_identity('2')};
  auto weights = bookmark_identity(reweighted, bundle);
  ASSERT_TRUE(weights.ok());
  EXPECT_FALSE([weights.get() isEqual:identity.get()]);
  Manifest renamed = manifest;
  renamed.bundle_digests =
      @{@"other.aimodel" : manifest.bundle_digests[bundle]};
  auto bundle_name = bookmark_identity(renamed, @"other.aimodel");
  ASSERT_TRUE(bundle_name.ok());
  EXPECT_FALSE([bundle_name.get() isEqual:identity.get()]);
  Manifest compiled = manifest;
  compiled.aot_compiled = true;
  auto kind = bookmark_identity(compiled, bundle);
  ASSERT_TRUE(kind.ok());
  EXPECT_FALSE([kind.get() isEqual:identity.get()]);
}

TEST_F(
    CoreAIBookmarkIdentityTest,
    IdentityRejectsUnselectedAndEscapingBundles) {
  EXPECT_FALSE(bookmark_identity(manifest, @"other.aimodel").ok());
  Manifest escaped = manifest;
  escaped.bundle_digests = @{@"../escape" : fixture_identity('1')};
  EXPECT_FALSE(bookmark_identity(escaped, @"../escape").ok());
}

TEST_F(CoreAIBookmarkRawTest, EnsurePublishesFullKeyFieldsReadableAsIdentity) {
  ASSERT_EQ(
      ensure_bookmark_identity(*locked, manifest, @"macOS", @"arch_b"),
      Error::Ok);
  auto identity = read_bookmark_identity(*locked);
  ASSERT_TRUE(identity.ok());
  auto expected = bookmark_identity(manifest, manifest.path.lastPathComponent);
  ASSERT_TRUE(expected.ok());
  EXPECT_TRUE([identity.get() isEqual:expected.get()]);
  NSData* bytes = [NSData dataWithContentsOfURL:identity_url(root.url, key)];
  ASSERT_NE(bytes, nil);
  id parsed = [NSJSONSerialization JSONObjectWithData:bytes
                                              options:0
                                                error:nil];
  ASSERT_TRUE([parsed isKindOfClass:NSArray.class]);
  NSArray* fields = parsed;
  ASSERT_EQ(fields.count, 9u);
  EXPECT_TRUE([fields[0] isEqual:@"1"]);
  EXPECT_TRUE([fields[1]
      isEqual:manifest.bundle_digests[manifest.path.lastPathComponent]]);
  EXPECT_TRUE([fields[2] isEqual:@"source"]);
  EXPECT_TRUE([fields[3] isEqual:manifest.path.lastPathComponent]);
  EXPECT_TRUE([fields[4] isEqual:@"macOS"]);
  EXPECT_TRUE([fields[5] isEqual:@"arch_b"]);
  for (id field in fields) {
    EXPECT_TRUE([field isKindOfClass:NSString.class] && [field length] > 0);
  }
  EXPECT_TRUE(no_temporary_bookmarks(root.url));
}

TEST_F(CoreAIBookmarkRawTest, EnsureReusesMatchingRecordWithoutRepublishing) {
  ASSERT_EQ(
      ensure_bookmark_identity(*locked, manifest, @"macOS", @"arch_b"),
      Error::Ok);
  NSData* published =
      [NSData dataWithContentsOfURL:identity_url(root.url, key)];
  ASSERT_NE(published, nil);
  for (StorageOperation operation :
       {StorageOperation::Write, StorageOperation::Rename}) {
    SCOPED_TRACE(static_cast<int>(operation));
    StorageFaultScope fault(operation);
    EXPECT_EQ(
        ensure_bookmark_identity(*locked, manifest, @"macOS", @"arch_b"),
        Error::Ok);
    EXPECT_EQ(fault.hits, 0);
  }
  EXPECT_TRUE([[NSData dataWithContentsOfURL:identity_url(root.url, key)]
      isEqual:published]);
  EXPECT_TRUE(no_temporary_bookmarks(root.url));
}

TEST_F(CoreAIBookmarkRawTest, MissingIdentityReadsAsNil) {
  auto identity = read_bookmark_identity(*locked);
  ASSERT_TRUE(identity.ok());
  EXPECT_EQ(identity.get(), nil);
}

TEST_F(CoreAIBookmarkRawTest, EnsureRepairsCorruptAndMisattributedRecords) {
  NSURL* metadata = identity_url(root.url, key);
  NSData* garbage = [@"not metadata" dataUsingEncoding:NSUTF8StringEncoding];
  ASSERT_TRUE([garbage writeToURL:metadata atomically:NO]);
  ASSERT_EQ(
      ensure_bookmark_identity(*locked, manifest, @"macOS", @"arch_b"),
      Error::Ok);
  auto repaired = read_bookmark_identity(*locked);
  ASSERT_TRUE(repaired.ok());
  auto expected = bookmark_identity(manifest, manifest.path.lastPathComponent);
  ASSERT_TRUE(expected.ok());
  EXPECT_TRUE([repaired.get() isEqual:expected.get()]);
  Manifest reweighted = manifest;
  reweighted.bundle_digests =
      @{manifest.path.lastPathComponent : fixture_identity('2')};
  auto foreign_key = test_bookmark_key(reweighted);
  ASSERT_TRUE(foreign_key.ok());
  auto foreign = lock_bookmark(root.url.path, foreign_key.get());
  ASSERT_TRUE(foreign.ok());
  ASSERT_EQ(
      ensure_bookmark_identity(*foreign.get(), reweighted, @"macOS", @"arch_b"),
      Error::Ok);
  NSData* foreign_bytes =
      [NSData dataWithContentsOfURL:identity_url(root.url, foreign_key.get())];
  ASSERT_NE(foreign_bytes, nil);
  ASSERT_TRUE([foreign_bytes writeToURL:metadata atomically:NO]);
  ASSERT_EQ(
      ensure_bookmark_identity(*locked, manifest, @"macOS", @"arch_b"),
      Error::Ok);
  auto restored = read_bookmark_identity(*locked);
  ASSERT_TRUE(restored.ok());
  EXPECT_TRUE([restored.get() isEqual:expected.get()]);
}

TEST_F(CoreAIBookmarkRawTest, RejectsMalformedIdentityRecords) {
  NSString* bundle = manifest.path.lastPathComponent;
  NSString* digest = manifest.bundle_digests[bundle];
  NSArray* base = @[
    @"1",
    digest,
    @"source",
    bundle,
    @"macOS",
    @"arch_b",
    @"AIModelCache.default",
    @"default-options-v1",
    @"persistent"
  ];
  NSMutableArray<NSData*>* payloads = [NSMutableArray array];
  [payloads addObject:[@"not json" dataUsingEncoding:NSUTF8StringEncoding]];
  for (id document in @[
         @{@"version" : @"1"},
         @"metadata",
         [base subarrayWithRange:NSMakeRange(0, 8)],
         [base arrayByAddingObject:@"extra"]
       ]) {
    [payloads addObject:[NSJSONSerialization
                            dataWithJSONObject:document
                                       options:NSJSONWritingFragmentsAllowed
                                         error:nil]];
  }
  NSMutableArray* numeric = [base mutableCopy];
  numeric[1] = @1;
  NSMutableArray* empty = [base mutableCopy];
  empty[4] = @"";
  NSMutableArray* kind = [base mutableCopy];
  kind[2] = @"compiled";
  NSMutableArray* undigestable = [base mutableCopy];
  undigestable[1] = [@"" stringByPaddingToLength:64
                                      withString:@"z"
                                 startingAtIndex:0];
  NSMutableArray* nested = [base mutableCopy];
  nested[3] = @"a/b";
  NSMutableArray* retargeted = [base mutableCopy];
  retargeted[8] = @"other";
  for (NSArray* variant in
       @[ numeric, empty, kind, undigestable, nested, retargeted ]) {
    [payloads addObject:[NSJSONSerialization dataWithJSONObject:variant
                                                        options:0
                                                          error:nil]];
  }
  NSURL* metadata = identity_url(root.url, key);
  for (NSData* payload in payloads) {
    SCOPED_TRACE(payload.length);
    ASSERT_NE(payload, nil);
    ASSERT_TRUE([payload writeToURL:metadata atomically:NO]);
    EXPECT_EQ(
        read_bookmark_identity(*locked).error(), Error::InvalidExternalData);
    EXPECT_TRUE([[NSData dataWithContentsOfURL:metadata] isEqual:payload]);
  }
}

TEST_F(CoreAIBookmarkRawTest, RejectsOversizedIdentityRecord) {
  NSURL* metadata = identity_url(root.url, key);
  FileDescriptor file(open(
      metadata.fileSystemRepresentation,
      O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
      0600));
  ASSERT_GE(file.get(), 0);
  ASSERT_EQ(ftruncate(file.get(), 128 * 1024 + 1), 0);
  ASSERT_EQ(close(file.release()), 0);
  EXPECT_EQ(
      read_bookmark_identity(*locked).error(), Error::InvalidExternalData);
  struct stat info {};
  ASSERT_EQ(lstat(metadata.fileSystemRepresentation, &info), 0);
  EXPECT_EQ(info.st_size, static_cast<off_t>(128 * 1024 + 1));
}

TEST_F(CoreAIBookmarkRawTest, RejectsLinkedIdentityRecordWithoutReadingTarget) {
  NSURL* metadata = identity_url(root.url, key);
  NSURL* target = [root.url URLByAppendingPathComponent:@"target"];
  NSData* bytes = [@"target bytes" dataUsingEncoding:NSUTF8StringEncoding];
  ASSERT_TRUE([bytes writeToURL:target atomically:NO]);
  ASSERT_EQ(
      symlink(
          target.fileSystemRepresentation, metadata.fileSystemRepresentation),
      0);
  EXPECT_EQ(
      read_bookmark_identity(*locked).error(), Error::InvalidExternalData);
  EXPECT_TRUE(([[NSData dataWithContentsOfURL:target] isEqual:bytes]));
  struct stat info {};
  ASSERT_EQ(lstat(metadata.fileSystemRepresentation, &info), 0);
  EXPECT_TRUE(S_ISLNK(info.st_mode));
}

TEST_F(CoreAIBookmarkRawTest, RejectsDirectoryAndFifoIdentityRecords) {
  NSURL* metadata = identity_url(root.url, key);
  ASSERT_EQ(mkdir(metadata.fileSystemRepresentation, 0700), 0);
  EXPECT_EQ(
      read_bookmark_identity(*locked).error(), Error::InvalidExternalData);
  ASSERT_EQ(rmdir(metadata.fileSystemRepresentation), 0);
  ASSERT_EQ(mkfifo(metadata.fileSystemRepresentation, 0600), 0);
  EXPECT_EQ(
      read_bookmark_identity(*locked).error(), Error::InvalidExternalData);
}

TEST_F(CoreAIBookmarkRawTest, RemoveIdentityPreservesBookmarkAndStaging) {
  NSData* bytes = [@"bookmark" dataUsingEncoding:NSUTF8StringEncoding];
  ASSERT_EQ(write_bookmark(*locked, bytes), Error::Ok);
  ASSERT_EQ(
      ensure_bookmark_identity(*locked, manifest, @"macOS", @"arch_b"),
      Error::Ok);
  auto staging = prepare_bookmark_staging(*locked);
  ASSERT_TRUE(staging.ok());
  ASSERT_EQ(remove_bookmark_identity(*locked), Error::Ok);
  EXPECT_FALSE([NSFileManager.defaultManager
      fileExistsAtPath:identity_url(root.url, key).path]);
  EXPECT_TRUE(bookmark_bytes_equal(*locked, bytes));
  EXPECT_TRUE([NSFileManager.defaultManager fileExistsAtPath:staging.get()]);
  auto reread = read_bookmark_identity(*locked);
  ASSERT_TRUE(reread.ok());
  EXPECT_EQ(reread.get(), nil);
  EXPECT_EQ(remove_bookmark_identity(*locked), Error::Ok);
}

TEST_F(CoreAIBookmarkPathTest, EnsureRejectsSelectedOrContextOutsideHeldKey) {
  auto locked = lock_bookmark(root.url.path, key);
  ASSERT_TRUE(locked.ok());
  Manifest reweighted = manifest;
  reweighted.bundle_digests =
      @{manifest.path.lastPathComponent : fixture_identity('2')};
  EXPECT_EQ(
      ensure_bookmark_identity(*locked.get(), reweighted, @"macOS", @"arch_b"),
      Error::InvalidArgument);
  EXPECT_EQ(
      ensure_bookmark_identity(*locked.get(), manifest, @"iOS", @"arch_b"),
      Error::InvalidArgument);
  EXPECT_EQ(
      ensure_bookmark_identity(*locked.get(), manifest, @"macOS", @"arch_a"),
      Error::InvalidArgument);
  EXPECT_FALSE([NSFileManager.defaultManager
      fileExistsAtPath:identity_url(root.url, key).path]);
}

TEST_F(CoreAIBookmarkPathTest, EnsureRejectsStagingOnlyLock) {
  NSURL* staged = [[root.url URLByAppendingPathComponent:@"staging"]
      URLByAppendingPathComponent:key];
  ASSERT_TRUE([NSFileManager.defaultManager createDirectoryAtURL:staged
                                     withIntermediateDirectories:YES
                                                      attributes:nil
                                                           error:nil]);
  auto existing = lock_bookmark(root.url.path, key, false);
  ASSERT_TRUE(existing.ok());
  ASSERT_NE(existing.get(), nullptr);
  EXPECT_EQ(
      ensure_bookmark_identity(*existing.get(), manifest, @"macOS", @"arch_b"),
      Error::InvalidArgument);
  EXPECT_FALSE([NSFileManager.defaultManager
      fileExistsAtPath:identity_url(root.url, key).path]);
}

TEST_F(CoreAIBookmarkPathTest, RemoveIdentityThroughStagingOnlyLockSucceeds) {
  NSURL* staged = [[root.url URLByAppendingPathComponent:@"staging"]
      URLByAppendingPathComponent:key];
  ASSERT_TRUE([NSFileManager.defaultManager createDirectoryAtURL:staged
                                     withIntermediateDirectories:YES
                                                      attributes:nil
                                                           error:nil]);
  auto existing = lock_bookmark(root.url.path, key, false);
  ASSERT_TRUE(existing.ok());
  ASSERT_NE(existing.get(), nullptr);
  EXPECT_EQ(remove_bookmark_identity(*existing.get()), Error::Ok);
}

TEST_F(CoreAIBookmarkPathTest, RejectsMetadataWrittenForAnotherKey) {
  Manifest reweighted = manifest;
  reweighted.bundle_digests =
      @{manifest.path.lastPathComponent : fixture_identity('2')};
  auto foreign_key = test_bookmark_key(reweighted);
  ASSERT_TRUE(foreign_key.ok());
  auto foreign = lock_bookmark(root.url.path, foreign_key.get());
  ASSERT_TRUE(foreign.ok());
  ASSERT_EQ(
      ensure_bookmark_identity(*foreign.get(), reweighted, @"macOS", @"arch_b"),
      Error::Ok);
  NSData* foreign_bytes =
      [NSData dataWithContentsOfURL:identity_url(root.url, foreign_key.get())];
  ASSERT_NE(foreign_bytes, nil);
  auto locked = lock_bookmark(root.url.path, key);
  ASSERT_TRUE(locked.ok());
  ASSERT_TRUE([foreign_bytes writeToURL:identity_url(root.url, key)
                             atomically:NO]);
  EXPECT_EQ(
      read_bookmark_identity(*locked.get()).error(),
      Error::InvalidExternalData);
  EXPECT_TRUE([[NSData dataWithContentsOfURL:identity_url(root.url, key)]
      isEqual:foreign_bytes]);
}

TEST_F(CoreAIBookmarkPathTest, MetadataOnlyEntriesListLockAndRemove) {
  auto locked = lock_bookmark(root.url.path, key);
  ASSERT_TRUE(locked.ok());
  ASSERT_EQ(
      ensure_bookmark_identity(*locked.get(), manifest, @"macOS", @"arch_b"),
      Error::Ok);
  locked.get().reset();
  auto inventory = inventory_bookmark_keys(root.url.path);
  ASSERT_TRUE(inventory.ok());
  EXPECT_TRUE([inventory.get() containsObject:key]);
  auto existing = lock_bookmark(root.url.path, key, false);
  ASSERT_TRUE(existing.ok());
  ASSERT_NE(existing.get(), nullptr);
  auto identity = read_bookmark_identity(*existing.get());
  ASSERT_TRUE(identity.ok());
  auto expected = bookmark_identity(manifest, manifest.path.lastPathComponent);
  ASSERT_TRUE(expected.ok());
  EXPECT_TRUE([identity.get() isEqual:expected.get()]);
  EXPECT_TRUE(bookmark_bytes_equal(*existing.get(), nil));
  ASSERT_EQ(remove_bookmark_identity(*existing.get()), Error::Ok);
  EXPECT_FALSE([NSFileManager.defaultManager
      fileExistsAtPath:identity_url(root.url, key).path]);
  auto relisted = inventory_bookmark_keys(root.url.path);
  ASSERT_TRUE(relisted.ok());
  EXPECT_FALSE([relisted.get() containsObject:key]);
  auto absent = lock_bookmark(root.url.path, key, false);
  ASSERT_TRUE(absent.ok());
  EXPECT_EQ(absent.get(), nullptr);
}

} // namespace
