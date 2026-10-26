/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>

#include <executorch/runtime/core/data_loader.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/core/portable_type/device.h>
#include <executorch/runtime/core/result.h>

namespace executorch {
namespace runtime {

/**
 * Abstract interface for device-specific memory allocation.
 *
 * Each device type (CUDA, etc.) provides a concrete implementation
 * that handles memory allocation on that device. Implementations are
 * expected to be singletons with static lifetime, registered via
 * DeviceAllocatorRegistry.
 */
class DeviceAllocator {
 public:
  /**
   * Default alignment of memory returned by allocate(). Reuses
   * MemoryAllocator::kDefaultAlignment so host- and device-side allocations
   * share the same baseline contract. Backends whose underlying device APIs
   * already provide stronger guarantees (e.g. cudaMalloc returns 256-byte
   * aligned pointers) will trivially satisfy this.
   */
  static constexpr size_t kDefaultAlignment =
      MemoryAllocator::kDefaultAlignment;

  virtual ~DeviceAllocator() = default;
  /**
   * Allocate device memory.
   *
   * @param nbytes Number of bytes to allocate.
   * @param index The device index.
   * @param alignment Minimum alignment of the returned pointer in bytes.
   *     Must be a power of 2. Defaults to kDefaultAlignment.
   * @return A Result containing the device pointer on success, or an error.
   */
  virtual Result<void*> allocate(
      size_t nbytes,
      etensor::DeviceIndex index,
      size_t alignment = kDefaultAlignment) = 0;

  /**
   * Deallocate device memory previously allocated via allocate().
   *
   * @param ptr Pointer to the memory to deallocate.
   * @param index The device index.
   */
  virtual void deallocate(void* ptr, etensor::DeviceIndex index) = 0;

  /**
   * Copy data from host memory to device memory.
   *
   * @param dst Destination pointer (device memory).
   * @param src Source pointer (host memory).
   * @param nbytes Number of bytes to copy.
   * @param index The device index.
   * @return Error::Ok on success, or an appropriate error code on failure.
   */
  virtual Error copy_host_to_device(
      void* dst,
      const void* src,
      size_t nbytes,
      etensor::DeviceIndex index) = 0;

  /**
   * Copy data from device memory to host memory.
   *
   * @param dst Destination pointer (host memory).
   * @param src Source pointer (device memory).
   * @param nbytes Number of bytes to copy.
   * @param index The device index.
   * @return Error::Ok on success, or an appropriate error code on failure.
   */
  virtual Error copy_device_to_host(
      void* dst,
      const void* src,
      size_t nbytes,
      etensor::DeviceIndex index) = 0;

  /**
   * Returns the device type this allocator handles.
   */
  virtual etensor::DeviceType device_type() const = 0;

  /**
   * Waits until the copies to device `index` that this allocator made on the
   * calling thread have finished, so that work on any stream of the device can
   * read their results.
   * The default does nothing, which suits allocators whose copies finish
   * before they return.
   *
   * @param index The device index.
   * @return Error::Ok on success, or an appropriate error code on failure.
   */
  virtual Error synchronize(etensor::DeviceIndex index) {
    (void)index;
    return Error::Ok;
  }

  // The defaults below stay in this header on purpose. While this class has
  // no out-of-line virtual function, its vtable and these bodies are only
  // compiled where a concrete allocator is defined, so programs that never
  // use device memory do not carry them.

  /**
   * Copies the first `size` bytes of host `data` to `dst` on device `index`,
   * then waits like synchronize(). Takes a data loader's result as it is, so
   * a failed load can be passed straight in, and returns its error.
   *
   * The default uses copy_host_to_device() and synchronize(). An override
   * must keep the same contract: pass on the error of `data`, accept zero
   * bytes, reject a buffer shorter than `size`, and finish the copy before it
   * returns, because `data` is freed then.
   *
   * @param data The host data, as a data loader returns it.
   * @param dst The device memory to copy to.
   * @param size The number of bytes to copy. `data` must hold at least this
   *     many.
   * @param index The device index.
   * @return Error::Ok on success, or an appropriate error code on failure.
   */
  virtual Error copy_from_host(
      Result<FreeableBuffer> data,
      void* dst,
      size_t size,
      etensor::DeviceIndex index) {
    if (!data.ok()) {
      return data.error();
    }
    // A data loader returns no buffer for zero bytes, which allocators reject.
    if (size == 0) {
      return Error::Ok;
    }
    ET_CHECK_OR_RETURN_ERROR(
        data->size() >= size,
        InvalidProgram,
        "Loaded %zu bytes for %zu bytes of device memory",
        data->size(),
        size);
    ET_CHECK_OK_OR_RETURN_ERROR(
        copy_host_to_device(dst, data->data(), size, index));
    // The caller may read `dst` on a stream the copy is not ordered with, and
    // `data` is freed when this returns.
    return synchronize(index);
  }

  /**
   * Loads `size` bytes at `offset` of a data loader's data into `dst` on
   * device `index`, then waits like synchronize().
   *
   * The default loads the bytes on the host and passes them to
   * copy_from_host(). An allocator can override it to read the loader's data
   * straight into device memory. Keeping the host buffer in here also keeps
   * that code out of callers, so builds without device memory stay smaller.
   *
   * @param loader The data loader to read from.
   * @param offset The offset of the bytes in the loader's data.
   * @param size The number of bytes to load.
   * @param segment_info Describes the segment the bytes belong to.
   * @param dst The device memory to load into.
   * @param index The device index.
   * @return Error::Ok on success, or an appropriate error code on failure.
   */
  virtual Error load_segment_into(
      const DataLoader& loader,
      size_t offset,
      size_t size,
      const DataLoader::SegmentInfo& segment_info,
      void* dst,
      etensor::DeviceIndex index) {
    return copy_from_host(
        loader.load(offset, size, segment_info), dst, size, index);
  }
};

/**
 * Registry for device allocators.
 *
 * Provides a global mapping from DeviceType to DeviceAllocator instances.
 * Device allocators register themselves at static initialization time,
 * and the runtime queries the registry to find the appropriate allocator
 * for a given device type.
 *
 * Threading contract:
 * - Registration is expected to happen once per device type during static
 *   initialization (single-threaded). The registry itself does not perform
 *   any locking around register_allocator()/get_allocator(), and concurrent
 *   registration is not supported.
 * - After registration, get_allocator() is safe to call concurrently from
 *   multiple threads because the underlying array is never mutated again.
 * - The DeviceAllocator implementation is responsible for its own
 *   thread-safety. When multiple Programs are loaded concurrently and each
 *   needs device memory, the allocator must serialize access to any shared
 *   state internally (similar to how XNNPACK's weight cache guards its
 *   internal state). The registry does not provide any synchronization on
 *   behalf of the allocator.
 */
class DeviceAllocatorRegistry {
 public:
  /**
   * Returns the singleton instance of the registry.
   */
  static DeviceAllocatorRegistry& instance();

  /**
   * Register an allocator. The device type is taken from
   * alloc->device_type(). Each device type may only be registered once;
   * attempting to register a second allocator for the same device type
   * will abort.
   *
   * Not thread-safe. Expected to be called during static initialization.
   *
   * @param alloc Pointer to the allocator (must have static lifetime).
   */
  void register_allocator(DeviceAllocator* alloc);

  /**
   * Get the allocator for a specific device type.
   *
   * Safe to call concurrently with other get_allocator() calls.
   *
   * @param type The device type.
   * @return Pointer to the allocator, or nullptr if not registered.
   */
  DeviceAllocator* get_allocator(etensor::DeviceType type);

 private:
  DeviceAllocatorRegistry() = default;

  // Singletons must not be copied or moved; instance() returns a reference,
  // and silently shallow-copying the registry would lead to confusing bugs
  // where modifications to the copy don't affect the real singleton.
  DeviceAllocatorRegistry(const DeviceAllocatorRegistry&) = delete;
  DeviceAllocatorRegistry& operator=(const DeviceAllocatorRegistry&) = delete;
  DeviceAllocatorRegistry(DeviceAllocatorRegistry&&) = delete;
  DeviceAllocatorRegistry& operator=(DeviceAllocatorRegistry&&) = delete;

  // Fixed-size array indexed by device type. This avoids dynamic allocation
  // and is suitable for embedded environments.
  DeviceAllocator* allocators_[etensor::kNumDeviceTypes] = {};
};

// Convenience free functions

/**
 * Register a device allocator. The device type is taken from
 * alloc->device_type(). See DeviceAllocatorRegistry::register_allocator()
 * for the threading contract.
 *
 * @param alloc Pointer to the allocator (must have static lifetime).
 */
void register_device_allocator(DeviceAllocator* alloc);

/**
 * Get the device allocator for a specific device type.
 *
 * @param type The device type.
 * @return Pointer to the allocator, or nullptr if not registered.
 */
DeviceAllocator* get_device_allocator(etensor::DeviceType type);

} // namespace runtime
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::runtime::DeviceAllocator;
using ::executorch::runtime::DeviceAllocatorRegistry;
using ::executorch::runtime::get_device_allocator;
using ::executorch::runtime::register_device_allocator;
} // namespace executor
} // namespace torch
