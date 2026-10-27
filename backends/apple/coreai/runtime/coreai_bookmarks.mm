/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#import "coreai_bookmarks.h"
#import "coreai_storage.h"

#include <CommonCrypto/CommonDigest.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>

namespace executorch::backends::coreai {
namespace {
using runtime::Error;
using runtime::Result;
// An allocation bound, not an SDK bookmark format limit.
constexpr size_t kBookmarkByteLimit = 8 * 1024 * 1024;

bool text(id value) {
  return [value isKindOfClass:NSString.class] && [value length] > 0 &&
      [value length] <= 4096 &&
      [value rangeOfString:@"\0"].location == NSNotFound;
}
bool component(id value) {
  return text(value) && ![value containsString:@"/"] &&
      ![value containsString:@"\\"] && ![value isEqual:@"."] &&
      ![value isEqual:@".."];
}
bool digest(id value) {
  return text(value) && [value length] == 64 &&
      [value rangeOfCharacterFromSet:[[NSCharacterSet
                                         characterSetWithCharactersInString:
                                             @"0123456789abcdef"] invertedSet]]
          .location == NSNotFound;
}
NSString* bookmark_name(NSString* key) {
  return [key stringByAppendingString:@".bookmark"];
}

constexpr size_t kIdentityByteLimit = 128 * 1024;
NSString* identity_name(NSString* key) {
  return [key stringByAppendingString:@".metadata"];
}

Result<NSArray<NSString*>*> identity_fields(
    const Manifest& selected,
    NSString* bundle) {
  ET_CHECK_OR_RETURN_ERROR(
      component(bundle),
      DelegateInvalidCompatibility,
      "Invalid Core AI bundle name");
  NSString* exported_digest = selected.bundle_digests[bundle];
  ET_CHECK_OR_RETURN_ERROR(
      digest(exported_digest),
      DelegateInvalidCompatibility,
      "Missing Core AI bundle digest");
  return @[
    @"1",
    exported_digest,
    selected.aot_compiled ? @"aot" : @"source",
    bundle
  ];
}

Result<NSArray<NSString*>*> key_fields(
    const Manifest& selected,
    NSString* platform,
    NSString* architecture) {
  auto identity = identity_fields(selected, selected.path.lastPathComponent);
  if (!identity.ok())
    return identity.error();
  ET_CHECK_OR_RETURN_ERROR(
      text(platform) && text(architecture),
      DelegateInvalidCompatibility,
      "Missing Core AI cache context");
  return [identity.get() arrayByAddingObjectsFromArray:@[
    platform,
    architecture,
    @"AIModelCache.default",
    @"default-options-v1",
    @"persistent"
  ]];
}

Result<NSString*> hash_fields(NSArray<NSString*>* fields) {
  NSData* encoded = [NSJSONSerialization dataWithJSONObject:fields
                                                    options:0
                                                      error:nil];
  ET_CHECK_OR_RETURN_ERROR(
      encoded != nil, InvalidExternalData, "Cannot encode bookmark key");
  unsigned char bytes[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256(encoded.bytes, static_cast<CC_LONG>(encoded.length), bytes);
  NSMutableString* key = [NSMutableString stringWithCapacity:64];
  for (unsigned char byte : bytes)
    [key appendFormat:@"%02x", byte];
  return [key copy];
}

} // namespace

Result<NSString*> bookmark_identity(
    const Manifest& manifest,
    NSString* bundle) {
  auto fields = identity_fields(manifest, bundle);
  if (!fields.ok())
    return fields.error();
  return hash_fields(fields.get());
}

Result<NSString*> bookmark_key(
    const Manifest& selected,
    NSString* platform,
    NSString* architecture) {
  auto fields = key_fields(selected, platform, architecture);
  if (!fields.ok())
    return fields.error();
  return hash_fields(fields.get());
}

namespace {
// Returns -1 when the root does not exist.
Result<int> open_root(NSString* root) {
  const int fd = retry_eintr([&] {
    return open(
        root.fileSystemRepresentation, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  });
  if (fd < 0 && errno == ENOENT)
    return -1;
  ET_CHECK_OR_RETURN_ERROR(
      fd >= 0, AccessFailed, "Cannot open Core AI assets root");
  return fd;
}

// Returns -1 when the child does not exist and create is false.
Result<int> open_child(int root_fd, NSString* name, bool create) {
  bool created = false;
  if (create) {
    const int made = retry_eintr(
        [&] { return mkdirat(root_fd, name.fileSystemRepresentation, 0700); });
    ET_CHECK_OR_RETURN_ERROR(
        made == 0 || errno == EEXIST,
        AccessFailed,
        "Cannot create Core AI assets directory");
    created = made == 0;
  }
  const int fd = retry_eintr([&] {
    return openat(
        root_fd,
        name.fileSystemRepresentation,
        O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
  });
  if (fd < 0 && errno == ENOENT && !create)
    return -1;
  ET_CHECK_OR_RETURN_ERROR(
      fd >= 0, AccessFailed, "Invalid Core AI assets directory");
  if (created) {
    FileDescriptor child(fd);
    ET_CHECK_OK_OR_RETURN_ERROR(sync_storage_directory(root_fd));
    return child.release();
  }
  return fd;
}

Result<bool> entry_exists(int root_fd, NSString* key) {
  for (NSString* name in @[ @"bookmarks", @"staging" ]) {
    auto opened = open_child(root_fd, name, false);
    if (!opened.ok())
      return opened.error();
    FileDescriptor child(opened.get());
    if (child.get() < 0)
      continue;
    NSArray<NSString*>* entries = [name isEqual:@"bookmarks"]
        ? @[ bookmark_name(key), identity_name(key) ]
        : @[ key ];
    for (NSString* entry in entries) {
      struct stat info;
      const int result = retry_eintr([&] {
        return fstatat(
            child.get(),
            entry.fileSystemRepresentation,
            &info,
            AT_SYMLINK_NOFOLLOW);
      });
      if (result == 0)
        return true;
      ET_CHECK_OR_RETURN_ERROR(
          errno == ENOENT, AccessFailed, "Cannot inspect Core AI cache entry");
    }
  }
  return false;
}
} // namespace

Result<NSString*> resolve_bookmark_root(NSString* path, bool create) {
  ET_CHECK_OR_RETURN_ERROR(
      text(path) && [path hasPrefix:@"/"],
      InvalidArgument,
      "Core AI assets require an absolute root path");
  if (create) {
    return prepare_storage_root(path, true);
  }
  struct stat info;
  const int result =
      retry_eintr([&] { return stat(path.fileSystemRepresentation, &info); });
  if (result < 0 && errno == ENOENT)
    return path;
  ET_CHECK_OR_RETURN_ERROR(
      result == 0 && S_ISDIR(info.st_mode),
      AccessFailed,
      "Invalid Core AI assets root");
  return path;
}

Result<NSArray<NSString*>*> inventory_bookmark_keys(NSString* root) {
  auto resolved = resolve_bookmark_root(root, false);
  if (!resolved.ok())
    return resolved.error();
  auto opened = open_root(resolved.get());
  if (!opened.ok())
    return opened.error();
  FileDescriptor directory(opened.get());
  if (directory.get() < 0)
    return @[];
  NSMutableSet<NSString*>* keys = [NSMutableSet set];
  for (NSString* name in @[ @"bookmarks", @"staging" ]) {
    auto child_result = open_child(directory.get(), name, false);
    if (!child_result.ok())
      return child_result.error();
    FileDescriptor child(child_result.get());
    if (child.get() < 0)
      continue;
    auto children = storage_children(child.get(), true);
    if (!children.ok())
      return children.error();
    for (NSString* entry in children.get()) {
      NSString* key = entry;
      if ([name isEqual:@"bookmarks"]) {
        NSString* suffix =
            [entry hasSuffix:@".bookmark"] ? @".bookmark" : @".metadata";
        if (![entry hasSuffix:suffix])
          continue;
        key = [entry substringToIndex:entry.length - suffix.length];
      }
      if (digest(key))
        [keys addObject:key];
    }
  }
  return [keys.allObjects sortedArrayUsingSelector:@selector(compare:)];
}

BookmarkLock::BookmarkLock(
    int root,
    int bookmarks,
    int file,
    NSString* root_path,
    NSString* key)
    : root_(root),
      bookmarks_(bookmarks),
      file_(file),
      root_path_(root_path),
      key_(key) {}

Result<std::unique_ptr<BookmarkLock>>
lock_bookmark(NSString* root, NSString* key, bool create) {
  ET_CHECK_OR_RETURN_ERROR(
      digest(key), InvalidArgument, "Invalid Core AI bookmark key");
  auto resolved = resolve_bookmark_root(root, create);
  if (!resolved.ok())
    return resolved.error();
  root = resolved.get();
  auto opened = open_root(root);
  if (!opened.ok())
    return opened.error();
  FileDescriptor directory(opened.get());
  if (directory.get() < 0) {
    ET_CHECK_OR_RETURN_ERROR(
        !create, AccessFailed, "Core AI assets root disappeared");
    return std::unique_ptr<BookmarkLock>();
  }
  if (!create) {
    auto exists = entry_exists(directory.get(), key);
    if (!exists.ok())
      return exists.error();
    if (!exists.get())
      return std::unique_ptr<BookmarkLock>();
  }
  auto lock_result = open_child(directory.get(), @"locks", true);
  if (!lock_result.ok())
    return lock_result.error();
  FileDescriptor locks(lock_result.get());
  NSString* name = [key stringByAppendingString:@".lock"];
  FileDescriptor file(
      open_storage_shared_file(locks.get(), name.fileSystemRepresentation));
  struct stat info;
  ET_CHECK_OR_RETURN_ERROR(
      file.get() >= 0 && fstat(file.get(), &info) == 0 && S_ISREG(info.st_mode),
      AccessFailed,
      "Invalid Core AI bookmark lock");
  ET_CHECK_OR_RETURN_ERROR(
      storage_fault(StorageOperation::BeforeBookmarkLock) == 0,
      AccessFailed,
      "Cannot begin Core AI bookmark lock acquisition");
  ET_CHECK_OR_RETURN_ERROR(
      retry_eintr([&] { return flock(file.get(), LOCK_EX); }) == 0,
      AccessFailed,
      "Cannot lock Core AI bookmark");
  if (!create) {
    auto exists = entry_exists(directory.get(), key);
    if (!exists.ok())
      return exists.error();
    if (!exists.get())
      return std::unique_ptr<BookmarkLock>();
  }
  auto bookmark_result = open_child(directory.get(), @"bookmarks", create);
  if (!bookmark_result.ok())
    return bookmark_result.error();
  FileDescriptor bookmarks(bookmark_result.get());
  return std::unique_ptr<BookmarkLock>(new BookmarkLock(
      directory.release(),
      bookmarks.release(),
      file.release(),
      root,
      [key copy]));
}

namespace {
Result<NSData*>
read_record(int directory, NSString* name, size_t limit, bool maintenance) {
  if (directory < 0)
    return static_cast<NSData*>(nil);
  if (maintenance) {
    struct stat entry;
    const int inspected = retry_eintr([&] {
      return fstatat(
          directory,
          name.fileSystemRepresentation,
          &entry,
          AT_SYMLINK_NOFOLLOW);
    });
    if (inspected < 0 && errno == ENOENT)
      return static_cast<NSData*>(nil);
    ET_CHECK_OR_RETURN_ERROR(
        inspected == 0, AccessFailed, "Cannot inspect Core AI cache record");
    if (!S_ISREG(entry.st_mode))
      return Error::InvalidExternalData;
  }
  FileDescriptor file(retry_eintr([&] {
    return openat(
        directory,
        name.fileSystemRepresentation,
        O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC);
  }));
  if (file.get() < 0 && errno == ENOENT)
    return static_cast<NSData*>(nil);
  if (file.get() < 0 && errno == ELOOP && maintenance)
    return Error::InvalidExternalData;
  ET_CHECK_OR_RETURN_ERROR(
      file.get() >= 0, AccessFailed, "Cannot open Core AI bookmark");
  struct stat info;
  ET_CHECK_OR_RETURN_ERROR(
      fstat(file.get(), &info) == 0,
      AccessFailed,
      "Cannot inspect Core AI bookmark");
  ET_CHECK_OR_RETURN_ERROR(
      S_ISREG(info.st_mode) && info.st_size > 0 &&
          static_cast<uint64_t>(info.st_size) <= limit,
      InvalidExternalData,
      "Invalid Core AI bookmark file");
  NSMutableData* bytes = [NSMutableData dataWithLength:info.st_size];
  size_t offset = 0;
  while (offset < bytes.length) {
    ssize_t count = ::read(
        file.get(),
        static_cast<char*>(bytes.mutableBytes) + offset,
        bytes.length - offset);
    if (count < 0 && errno == EINTR)
      continue;
    ET_CHECK_OR_RETURN_ERROR(
        count > 0, AccessFailed, "Cannot read Core AI bookmark");
    offset += count;
  }
  return bytes;
}

} // namespace

Result<NSData*> read_bookmark(const BookmarkLock& lock, bool maintenance) {
  return read_record(
      lock.bookmarks_.get(),
      bookmark_name(lock.key_),
      kBookmarkByteLimit,
      maintenance);
}

Result<NSString*> read_bookmark_identity(const BookmarkLock& lock) {
  auto bytes = read_record(
      lock.bookmarks_.get(),
      identity_name(lock.key_),
      kIdentityByteLimit,
      true);
  if (!bytes.ok())
    return bytes.error();
  if (bytes.get() == nil)
    return static_cast<NSString*>(nil);
  id parsed = [NSJSONSerialization JSONObjectWithData:bytes.get()
                                              options:0
                                                error:nil];
  if (![parsed isKindOfClass:NSArray.class] || [parsed count] != 9)
    return Error::InvalidExternalData;
  NSArray<NSString*>* fields = parsed;
  for (id field in fields) {
    if (!text(field))
      return Error::InvalidExternalData;
  }
  if (![fields[2] isEqual:@"source"] && ![fields[2] isEqual:@"aot"])
    return Error::InvalidExternalData;
  Manifest selected;
  selected.path = fields[3];
  selected.bundle_digests = @{fields[3] : fields[1]};
  selected.aot_compiled = [fields[2] isEqual:@"aot"];
  auto expected = key_fields(selected, fields[4], fields[5]);
  if (!expected.ok() || ![expected.get() isEqual:fields])
    return Error::InvalidExternalData;
  auto key = hash_fields(fields);
  if (!key.ok())
    return key.error();
  if (![key.get() isEqual:lock.key_])
    return Error::InvalidExternalData;
  return bookmark_identity(selected, fields[3]);
}

Error ensure_bookmark_identity(
    const BookmarkLock& lock,
    const Manifest& selected,
    NSString* platform,
    NSString* architecture) {
  ET_CHECK_OR_RETURN_ERROR(
      lock.bookmarks_.get() >= 0,
      InvalidArgument,
      "Cannot attribute a staging-only lock");
  auto fields = key_fields(selected, platform, architecture);
  if (!fields.ok())
    return fields.error();
  auto key = hash_fields(fields.get());
  if (!key.ok())
    return key.error();
  ET_CHECK_OR_RETURN_ERROR(
      [key.get() isEqual:lock.key_],
      InvalidArgument,
      "Core AI cache identity does not match key");
  NSData* encoded = [NSJSONSerialization dataWithJSONObject:fields.get()
                                                    options:0
                                                      error:nil];
  ET_CHECK_OR_RETURN_ERROR(
      encoded != nil && encoded.length <= kIdentityByteLimit,
      InvalidExternalData,
      "Invalid Core AI cache identity");
  auto existing = read_record(
      lock.bookmarks_.get(),
      identity_name(lock.key_),
      kIdentityByteLimit,
      true);
  if (!existing.ok() && existing.error() != Error::InvalidExternalData)
    return existing.error();
  if (existing.ok() && [existing.get() isEqual:encoded]) {
    // A prior publication may have failed its post-rename flush.
    return sync_storage_directory(lock.bookmarks_.get(), true);
  }
  return publish_storage_data(
      lock.bookmarks_.get(), identity_name(lock.key_), encoded);
}

Error remove_bookmark_identity(const BookmarkLock& lock) {
  if (lock.bookmarks_.get() < 0)
    return Error::Ok;
  return remove_storage_entry(lock.bookmarks_.get(), identity_name(lock.key_));
}

Error write_bookmark(const BookmarkLock& lock, NSData* data) {
  ET_CHECK_OR_RETURN_ERROR(
      [data isKindOfClass:NSData.class] && data.length > 0 &&
          data.length <= kBookmarkByteLimit,
      InvalidExternalData,
      "Invalid Core AI bookmark data");
  ET_CHECK_OR_RETURN_ERROR(
      lock.bookmarks_.get() >= 0,
      InvalidArgument,
      "Cannot write through a staging-only Core AI bookmark lock");
  return publish_storage_data(
      lock.bookmarks_.get(), bookmark_name(lock.key_), data);
}

Error remove_bookmark(const BookmarkLock& lock) {
  if (lock.bookmarks_.get() < 0)
    return Error::Ok;
  return remove_storage_entry(lock.bookmarks_.get(), bookmark_name(lock.key_));
}

Error remove_bookmark_staging(
    const BookmarkLock& lock,
    bool remove,
    StorageRemovalMode mode) {
  return remove_storage_staging(lock.root_.get(), lock.key_, remove, mode);
}

Result<NSString*> prepare_bookmark_staging(const BookmarkLock& lock) {
  auto staging = open_child(lock.root_.get(), @"staging", true);
  if (!staging.ok())
    return staging.error();
  FileDescriptor directory(staging.get());
  return [lock.root_path_ stringByAppendingPathComponent:@"staging"];
}

} // namespace executorch::backends::coreai
