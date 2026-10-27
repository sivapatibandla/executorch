/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "coreai_storage.h"
#include "coreai_file.h"

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>

namespace executorch::backends::coreai {
namespace {
using runtime::Error;
using runtime::Result;
} // namespace

Error ensure_excluded_from_backup(NSURL* directory) {
  NSURL* fresh = [NSURL fileURLWithPath:directory.path isDirectory:YES];
  NSNumber* excluded = nil;
  NSError* error = nil;
  ET_CHECK_OR_RETURN_ERROR(
      [fresh getResourceValue:&excluded
                       forKey:NSURLIsExcludedFromBackupKey
                        error:&error],
      AccessFailed,
      "Cannot read Core AI backup exclusion: %s",
      error.localizedDescription.UTF8String);
  if (excluded.boolValue) {
    return Error::Ok;
  }
  ET_CHECK_OR_RETURN_ERROR(
      [fresh setResourceValue:@YES
                       forKey:NSURLIsExcludedFromBackupKey
                        error:&error],
      AccessFailed,
      "Cannot exclude Core AI storage from backup: %s",
      error.localizedDescription.UTF8String);
  return Error::Ok;
}

Result<NSString*> default_coreai_assets_root() {
  NSURL* caches =
      [NSFileManager.defaultManager URLsForDirectory:NSCachesDirectory
                                           inDomains:NSUserDomainMask]
          .firstObject;
  ET_CHECK_OR_RETURN_ERROR(
      caches.isFileURL, AccessFailed, "Cannot resolve Core AI cache directory");
  return [caches.path stringByAppendingPathComponent:@"executorch_coreai"];
}

Result<NSString*> prepare_storage_root(NSString* path, bool create) {
  ET_CHECK_OR_RETURN_ERROR(
      [path isKindOfClass:NSString.class] && [path hasPrefix:@"/"] &&
          [path rangeOfString:@"\0"].location == NSNotFound,
      InvalidArgument,
      "Core AI storage requires an absolute path");
  struct stat info;
  const int result =
      retry_eintr([&] { return stat(path.fileSystemRepresentation, &info); });
  ET_CHECK_OR_RETURN_ERROR(
      result == 0 || errno == ENOENT || errno == ENOTDIR,
      AccessFailed,
      "Cannot inspect Core AI storage directory");
  ET_CHECK_OR_RETURN_ERROR(
      result == 0 ? S_ISDIR(info.st_mode) : create && errno == ENOENT,
      InvalidArgument,
      "Core AI storage must be a directory");
  if (create) {
    NSError* error = nil;
    ET_CHECK_OR_RETURN_ERROR(
        [NSFileManager.defaultManager
                  createDirectoryAtPath:path
            withIntermediateDirectories:YES
                             attributes:@{
                               NSFilePosixPermissions : @0700
                             }
                                  error:&error],
        AccessFailed,
        "Cannot create Core AI storage directory: %s",
        error.localizedDescription.UTF8String);
    ET_CHECK_OK_OR_RETURN_ERROR(ensure_excluded_from_backup(
        [NSURL fileURLWithPath:path isDirectory:YES]));
  }
  return path;
}

#if defined(COREAI_ASSETS_TESTING) && COREAI_ASSETS_TESTING
namespace testing {
thread_local StorageFaultCallback storage_fault_callback = nullptr;
}

int storage_fault(StorageOperation operation) {
  return testing::storage_fault_callback == nullptr
      ? 0
      : testing::storage_fault_callback(operation);
}
#endif

namespace {
bool storage_component(NSString* name) {
  return [name isKindOfClass:NSString.class] && name.length > 0 &&
      ![name isEqualToString:@"."] && ![name isEqualToString:@".."] &&
      ![name containsString:@"/"] && ![name containsString:@"\0"];
}

Error write_file(
    int root_fd,
    NSString* relative_path,
    const void* data,
    size_t size,
    bool full_sync) {
  NSArray<NSString*>* components =
      [relative_path componentsSeparatedByString:@"/"];
  ET_CHECK_OR_RETURN_ERROR(
      components.count > 0 && (data != nullptr || size == 0),
      InvalidArgument,
      "Invalid Core AI storage write");
  for (NSString* component in components) {
    ET_CHECK_OR_RETURN_ERROR(
        storage_component(component), InvalidArgument, "Invalid storage path");
  }
  FileDescriptor parent(
      retry_eintr([&] { return fcntl(root_fd, F_DUPFD_CLOEXEC, 0); }));
  ET_CHECK_OR_RETURN_ERROR(
      parent.get() >= 0, AccessFailed, "Cannot open storage parent");
  for (NSUInteger i = 0; i + 1 < components.count; ++i) {
    const char* component = components[i].fileSystemRepresentation;
    ET_CHECK_OR_RETURN_ERROR(
        retry_eintr([&] { return mkdirat(parent.get(), component, 0700); }) ==
                0 ||
            errno == EEXIST,
        AccessFailed,
        "Cannot create Core AI source directory");
    FileDescriptor next(retry_eintr([&] {
      return openat(
          parent.get(),
          component,
          O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    }));
    ET_CHECK_OR_RETURN_ERROR(
        next.get() >= 0, AccessFailed, "Cannot open source directory");
    ET_CHECK_OK_OR_RETURN_ERROR(sync_storage_directory(parent.get()));
    parent.reset(next.release());
  }
  FileDescriptor file(retry_eintr([&] {
    return openat(
        parent.get(),
        components.lastObject.fileSystemRepresentation,
        O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC,
        0600);
  }));
  ET_CHECK_OR_RETURN_ERROR(
      file.get() >= 0, AccessFailed, "Cannot create Core AI storage file");
  const auto result = [&]() -> Error {
    const auto* bytes = static_cast<const uint8_t*>(data);
    while (size > 0) {
      const ssize_t count = storage_call(StorageOperation::Write, [&] {
        return write(file.get(), bytes, std::min(size, size_t(1024 * 1024)));
      });
      ET_CHECK_OR_RETURN_ERROR(
          count > 0, AccessFailed, "Cannot write Core AI storage file");
      bytes += count;
      size -= count;
    }
    ET_CHECK_OR_RETURN_ERROR(
        storage_call(
            StorageOperation::SyncFile, [&] { return fsync(file.get()); }) ==
                0 &&
            (!full_sync ||
             storage_call(
                 StorageOperation::FullSync,
                 [&] { return fcntl(file.get(), F_FULLFSYNC); }) == 0),
        AccessFailed,
        "Cannot synchronize Core AI file");
    ET_CHECK_OR_RETURN_ERROR(
        ::close(file.release()) == 0,
        AccessFailed,
        "Cannot close Core AI storage file");
    return sync_storage_directory(parent.get());
  }();
  if (result != Error::Ok) {
    if (retry_eintr([&] {
          return unlinkat(
              parent.get(), components.lastObject.fileSystemRepresentation, 0);
        }) == 0) {
      (void)sync_storage_directory(parent.get());
    } else {
      ET_LOG(
          Error, "Cannot remove failed Core AI storage write: errno=%d", errno);
    }
  }
  return result;
}

// Preflight the whole tree before unlinking anything; never follow source
// links.
Error visit_source_tree(int fd, bool remove) {
  auto names = storage_children(fd);
  if (!names.ok())
    return names.error();
  for (NSString* name in names.get()) {
    struct stat info;
    ET_CHECK_OR_RETURN_ERROR(
        retry_eintr([&] {
          return fstatat(
              fd, name.fileSystemRepresentation, &info, AT_SYMLINK_NOFOLLOW);
        }) == 0,
        AccessFailed,
        "Cannot inspect Core AI source entry");
    ET_CHECK_OR_RETURN_ERROR(
        S_ISDIR(info.st_mode) || S_ISREG(info.st_mode),
        InvalidExternalData,
        "Unexpected Core AI source entry type");
    if (S_ISDIR(info.st_mode)) {
      FileDescriptor child(retry_eintr([&] {
        return openat(
            fd,
            name.fileSystemRepresentation,
            O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
      }));
      ET_CHECK_OR_RETURN_ERROR(
          child.get() >= 0,
          AccessFailed,
          "Cannot open Core AI source directory");
      ET_CHECK_OK_OR_RETURN_ERROR(visit_source_tree(child.get(), remove));
    }
    if (remove) {
      ET_CHECK_OR_RETURN_ERROR(
          storage_call(
              StorageOperation::Remove,
              [&] {
                return unlinkat(
                    fd,
                    name.fileSystemRepresentation,
                    S_ISDIR(info.st_mode) ? AT_REMOVEDIR : 0);
              }) == 0,
          AccessFailed,
          "Cannot remove Core AI source entry");
    }
  }
  return remove ? sync_storage_directory(fd) : Error::Ok;
}
} // namespace

Result<NSArray<NSString*>*> storage_children(int fd, bool skip_invalid_names) {
  // Reopening avoids sharing the caller's directory offset.
  FileDescriptor copy(retry_eintr(
      [&] { return openat(fd, ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC); }));
  ET_CHECK_OR_RETURN_ERROR(
      copy.get() >= 0, AccessFailed, "Cannot open storage directory");
  DIR* directory = fdopendir(copy.get());
  ET_CHECK_OR_RETURN_ERROR(
      directory != nullptr,
      AccessFailed,
      "Cannot open storage directory stream");
  copy.release();
  NSMutableArray<NSString*>* names = [NSMutableArray array];
  Error result = Error::Ok;
  while (true) {
    errno = 0;
    dirent* entry = readdir(directory);
    if (entry == nullptr) {
      if (errno == EINTR)
        continue;
      if (errno != 0)
        result = Error::AccessFailed;
      break;
    }
    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
      continue;
    }
    NSString* name = [[NSString alloc] initWithBytes:entry->d_name
                                              length:strlen(entry->d_name)
                                            encoding:NSUTF8StringEncoding];
    if (name == nil) {
      if (skip_invalid_names) {
        continue;
      }
      result = Error::InvalidExternalData;
      break;
    }
    [names addObject:name];
  }
  const int closed = closedir(directory);
  ET_CHECK_OK_OR_RETURN_ERROR(result);
  ET_CHECK_OR_RETURN_ERROR(
      closed == 0, AccessFailed, "Cannot close storage directory stream");
  return [names sortedArrayUsingSelector:@selector(compare:)];
}

int open_storage_shared_file(int parent_fd, const char* name) {
  // APFS can return ENOENT to a concurrent O_CREAT loser without O_EXCL.
  int fd = retry_eintr([&] {
    return openat(
        parent_fd,
        name,
        O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC,
        0600);
  });
  if (fd < 0 && errno == EEXIST) {
    fd = retry_eintr([&] {
      return openat(parent_fd, name, O_RDWR | O_NOFOLLOW | O_CLOEXEC);
    });
  }
  return fd;
}

Error sync_storage_directory(int fd, bool full_sync) {
  ET_CHECK_OR_RETURN_ERROR(
      storage_call(
          StorageOperation::SyncDirectory, [&] { return fsync(fd); }) == 0,
      AccessFailed,
      "Cannot synchronize Core AI directory");
  if (full_sync) {
    ET_CHECK_OR_RETURN_ERROR(
        storage_call(
            StorageOperation::FullSync,
            [&] { return fcntl(fd, F_FULLFSYNC); }) == 0,
        AccessFailed,
        "Cannot fully synchronize Core AI directory");
  }
  return Error::Ok;
}

Error remove_storage_staging(int root_fd, NSString* key) {
  ET_CHECK_OR_RETURN_ERROR(
      [key isKindOfClass:NSString.class] && key.length == 64 &&
          [key rangeOfCharacterFromSet:
                   [[NSCharacterSet
                       characterSetWithCharactersInString:@"0123456789abcdef"]
                       invertedSet]]
                  .location == NSNotFound,
      InvalidArgument,
      "Invalid Core AI staging key");
  FileDescriptor staging(retry_eintr([&] {
    return openat(
        root_fd, "staging", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
  }));
  if (staging.get() < 0 && errno == ENOENT)
    return Error::Ok;
  ET_CHECK_OR_RETURN_ERROR(
      staging.get() >= 0,
      AccessFailed,
      "Cannot open Core AI staging directory");
  struct stat info;
  const int inspected = retry_eintr([&] {
    return fstatat(
        staging.get(),
        key.fileSystemRepresentation,
        &info,
        AT_SYMLINK_NOFOLLOW);
  });
  if (inspected < 0 && errno == ENOENT)
    return Error::Ok;
  ET_CHECK_OR_RETURN_ERROR(
      inspected == 0,
      AccessFailed,
      "Cannot inspect Core AI keyed staging directory");
  ET_CHECK_OR_RETURN_ERROR(
      S_ISDIR(info.st_mode),
      InvalidExternalData,
      "Core AI keyed staging entry is not a directory");
  FileDescriptor source(retry_eintr([&] {
    return openat(
        staging.get(),
        key.fileSystemRepresentation,
        O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
  }));
  ET_CHECK_OR_RETURN_ERROR(
      source.get() >= 0,
      AccessFailed,
      "Cannot open Core AI keyed staging directory");
  ET_CHECK_OK_OR_RETURN_ERROR(visit_source_tree(source.get(), false));
  ET_CHECK_OK_OR_RETURN_ERROR(visit_source_tree(source.get(), true));
  ET_CHECK_OR_RETURN_ERROR(
      storage_call(
          StorageOperation::Remove,
          [&] {
            return unlinkat(
                staging.get(), key.fileSystemRepresentation, AT_REMOVEDIR);
          }) == 0,
      AccessFailed,
      "Cannot remove Core AI keyed staging directory");
  return sync_storage_directory(staging.get());
}

Error write_storage_file(
    int root_fd,
    NSString* relative_path,
    const void* data,
    size_t size) {
  return write_file(root_fd, relative_path, data, size, false);
}

Error publish_storage_data(int root_fd, NSString* name, NSData* data) {
  ET_CHECK_OR_RETURN_ERROR(
      storage_component(name) && data != nil,
      InvalidArgument,
      "Invalid Core AI storage publication");
  NSString* temporary =
      [@".tmp-" stringByAppendingString:NSUUID.UUID.UUIDString];
  // Bookmarks are small and costly to lose, so they get a full device flush.
  auto result = write_file(root_fd, temporary, data.bytes, data.length, true);
  if (result == Error::Ok && storage_call(StorageOperation::Rename, [&] {
                               return renameat(
                                   root_fd,
                                   temporary.fileSystemRepresentation,
                                   root_fd,
                                   name.fileSystemRepresentation);
                             }) != 0) {
    ET_LOG(Error, "Cannot publish Core AI storage data: errno=%d", errno);
    result = Error::AccessFailed;
  }
  if (result == Error::Ok) {
    // Flush the renamed entry as well as the file's pre-rename contents.
    ET_CHECK_OR_RETURN_ERROR(
        storage_call(
            StorageOperation::FullSync,
            [&] { return fcntl(root_fd, F_FULLFSYNC); }) == 0,
        AccessFailed,
        "Cannot fully synchronize Core AI publication directory");
    return Error::Ok;
  }
  if (unlinkat(root_fd, temporary.fileSystemRepresentation, 0) == 0) {
    (void)sync_storage_directory(root_fd);
  }
  return result;
}

} // namespace executorch::backends::coreai
