#include <cub/block/block_scan.cuh>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

#define CUDA_CHECK(call)                                                                      \
  do                                                                                          \
  {                                                                                           \
    cudaError_t const status = (call);                                                        \
    if (status != cudaSuccess)                                                                \
      throw std::runtime_error(cudaGetErrorString(status));                                   \
  } while (false)

namespace
{
constexpr int block_threads = 256;
constexpr int blocks = 32768;
constexpr std::size_t full_items = static_cast<std::size_t>(block_threads) * blocks;
constexpr std::size_t partial_items = full_items - 63;
constexpr int warmups = 20;
constexpr int samples = 21;
constexpr int launches = 100;

template <class T>
__global__ __launch_bounds__(block_threads) void block_exclusive_sum(
  T* values, std::size_t items)
{
  using scan_t = cub::BlockScan<T, block_threads, cub::BLOCK_SCAN_WARP_SCANS>;
  __shared__ typename scan_t::TempStorage storage;

  std::size_t const tile_base = static_cast<std::size_t>(blockIdx.x) * block_threads;
  std::size_t const index = tile_base + threadIdx.x;
  int const valid = static_cast<int>(
    min(items - min(items, tile_base), static_cast<std::size_t>(block_threads)));
  T const input = threadIdx.x < valid ? values[index] : T{};
  T output{};
  scan_t(storage).ExclusiveSum(input, output);
  if (threadIdx.x < valid)
    values[index] = output;
}

template <class T>
void run(char const* type_name, std::size_t items)
{
  std::vector<T> host(items, T{1});
  T* baseline = nullptr;
  T* values = nullptr;
  CUDA_CHECK(cudaMalloc(&baseline, host.size() * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&values, host.size() * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(baseline, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));

  for (int iteration = 0; iteration < warmups; ++iteration)
  {
    CUDA_CHECK(cudaMemcpyAsync(values, baseline, host.size() * sizeof(T), cudaMemcpyDeviceToDevice));
    for (int launch = 0; launch < launches; ++launch)
      block_exclusive_sum<<<blocks, block_threads>>>(values, items);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start{}, end{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));
  std::vector<float> timings;
  timings.reserve(samples);
  for (int sample = 0; sample < samples; ++sample)
  {
    CUDA_CHECK(cudaMemcpyAsync(values, baseline, host.size() * sizeof(T), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaEventRecord(start));
    for (int launch = 0; launch < launches; ++launch)
      block_exclusive_sum<<<blocks, block_threads>>>(values, items);
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    float milliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, end));
    timings.push_back(milliseconds / launches);
  }

  CUDA_CHECK(cudaMemcpy(values, baseline, host.size() * sizeof(T), cudaMemcpyDeviceToDevice));
  block_exclusive_sum<<<blocks, block_threads>>>(values, items);
  CUDA_CHECK(cudaMemcpy(host.data(), values, host.size() * sizeof(T), cudaMemcpyDeviceToHost));
  for (std::size_t index = 0; index < items; ++index)
    if (host[index] != static_cast<T>(index % block_threads))
      throw std::runtime_error("CUB BlockScan result mismatch");

  std::sort(timings.begin(), timings.end());
  std::printf(
    "implementation=cccl algorithm=block_exclusive_sum type=%s threads=%d items=%zu blocks=%d launches=%d best_ms=%.6f median_ms=%.6f\n",
    type_name,
    block_threads,
    items,
    blocks,
    launches,
    timings.front(),
    timings[timings.size() / 2]);

  CUDA_CHECK(cudaEventDestroy(end));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaFree(values));
  CUDA_CHECK(cudaFree(baseline));
}
} // namespace

int main()
try
{
  run<std::uint32_t>("u32", full_items);
  run<std::uint32_t>("u32", partial_items);
  run<std::uint64_t>("u64", full_items);
  run<std::uint64_t>("u64", partial_items);
  return 0;
}
catch (std::exception const& error)
{
  std::fprintf(stderr, "CUB BlockScan reference failed: %s\n", error.what());
  return 1;
}
