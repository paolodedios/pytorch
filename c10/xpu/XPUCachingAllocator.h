#pragma once

#include <c10/core/AllocatorConfig.h>
#include <c10/core/CachingDeviceAllocator.h>
#include <c10/util/flat_hash_map.h>
#include <c10/xpu/XPUStream.h>
#include <vector>

namespace c10::xpu::XPUCachingAllocator {

namespace {
struct Block;
}
class DeviceCachingAllocator;
using DeviceStats = c10::CachingDeviceAllocator::DeviceStats;
class XPUAllocator : public DeviceAllocator {
 private:
  alignas(hardware_destructive_interference_size) std::mutex mutex;
  ska::flat_hash_map<void*, Block*> allocated_blocks;
  void add_allocated_block(Block* block);
  Block* get_allocated_block(void* ptr, bool remove = false);

 public:
  std::vector<std::unique_ptr<DeviceCachingAllocator>> device_allocators;
  void init(DeviceIndex device_count);
  bool initialized() override;
  void malloc(
      void** devPtr,
      DeviceIndex device,
      size_t size,
      sycl::queue& queue);
  void free(void* ptr);
  void emptyCache(MempoolId_t mempool_id) override;
  void recordStream(const DataPtr& ptr, c10::Stream stream) override;
  DataPtr allocate(size_t size) override;
  DeleterFnPtr raw_deleter() const override;
  void* raw_alloc(size_t size);
  void* raw_alloc_with_stream(size_t size, XPUStream stream);
  void raw_delete(void* ptr);
  void copy_data(void* dest, const void* src, std::size_t count) const final;
  void assertValidDevice(DeviceIndex device);
  DeviceStats getDeviceStats(DeviceIndex device) override;
  void resetPeakStats(DeviceIndex device) override;
  void resetAccumulatedStats(DeviceIndex device) override;
};

C10_XPU_API Allocator* get();

C10_XPU_API void init(DeviceIndex device_count);

C10_XPU_API void emptyCache();

C10_XPU_API void resetPeakStats(DeviceIndex device);

C10_XPU_API void resetAccumulatedStats(DeviceIndex device);

C10_XPU_API c10::CachingDeviceAllocator::DeviceStats getDeviceStats(
    DeviceIndex device);

C10_XPU_API void* raw_alloc(size_t size);

C10_XPU_API void raw_delete(void* ptr);

C10_XPU_API void recordStream(const DataPtr& dataPtr, XPUStream stream);

C10_XPU_API double getMemoryFraction(DeviceIndex device);

C10_XPU_API void setMemoryFraction(double fraction, DeviceIndex device);

} // namespace c10::xpu::XPUCachingAllocator
