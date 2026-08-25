#include <cub/warp/warp_merge_sort.cuh>

#include <cuda_runtime.h>
#include <cuda/std/functional>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#define CUDA_CHECK(call)                                                                       \
  do                                                                                           \
  {                                                                                            \
    const cudaError_t status = (call);                                                         \
    if (status != cudaSuccess)                                                                 \
      throw std::runtime_error(cudaGetErrorString(status));                                    \
  } while (false)

constexpr int items_per_thread = 4;
constexpr int logical_warp_threads = 32;
constexpr int threads_per_block = 256;
constexpr int warps_per_block = threads_per_block / logical_warp_threads;
constexpr int items_per_warp = items_per_thread * logical_warp_threads;
constexpr int items_per_block = threads_per_block * items_per_thread;

__global__ void fill_descending_tiles(std::uint32_t* data, std::uint64_t count)
{
  const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count)
    data[index] = items_per_warp - 1 - (index % items_per_warp);
}

__global__ void sort_tiles(std::uint32_t* data)
{
  using warp_sort_t = cub::WarpMergeSort<std::uint32_t, items_per_thread>;
  __shared__ typename warp_sort_t::TempStorage storage[warps_per_block];

  const int warp_id = threadIdx.x / logical_warp_threads;
  const int linear_tid = threadIdx.x % logical_warp_threads;
  const std::uint64_t thread_offset =
    static_cast<std::uint64_t>(blockIdx.x) * items_per_block +
    warp_id * items_per_warp + linear_tid * items_per_thread;
  std::uint32_t keys[items_per_thread];
#pragma unroll
  for (int item = 0; item < items_per_thread; ++item)
    keys[item] = data[thread_offset + item];

  warp_sort_t(storage[warp_id]).Sort(keys, cuda::std::less<>{});

#pragma unroll
  for (int item = 0; item < items_per_thread; ++item)
    data[thread_offset + item] = keys[item];
}

__global__ void validate_tiles(const std::uint32_t* data, std::uint64_t count, unsigned long long* errors)
{
  const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count && data[index] != index % items_per_warp)
    atomicAdd(errors, 1ull);
}

int main(int argc, char** argv)
try
{
  const std::uint64_t requested = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : (1ull << 24);
  const int iterations = argc > 2 ? std::atoi(argv[2]) : 50;
  const std::uint64_t count = requested / items_per_block * items_per_block;
  if (count == 0 || iterations <= 0)
    throw std::runtime_error("elements must cover one block and iterations must be nonzero");

  std::uint32_t* data = nullptr;
  CUDA_CHECK(cudaMalloc(&data, count * sizeof(*data)));
  cudaStream_t stream{};
  CUDA_CHECK(cudaStreamCreate(&stream));
  const int fill_threads = 256;
  const int fill_blocks = static_cast<int>((count + fill_threads - 1) / fill_threads);
  const int sort_blocks = static_cast<int>(count / items_per_block);

  for (int iteration = 0; iteration < 5; ++iteration)
  {
    fill_descending_tiles<<<fill_blocks, fill_threads, 0, stream>>>(data, count);
    sort_tiles<<<sort_blocks, threads_per_block, 0, stream>>>(data);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaEvent_t start{}, end{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));
  std::vector<double> timings;
  timings.reserve(iterations);
  for (int iteration = 0; iteration < iterations; ++iteration)
  {
    fill_descending_tiles<<<fill_blocks, fill_threads, 0, stream>>>(data, count);
    CUDA_CHECK(cudaEventRecord(start, stream));
    sort_tiles<<<sort_blocks, threads_per_block, 0, stream>>>(data);
    CUDA_CHECK(cudaEventRecord(end, stream));
    CUDA_CHECK(cudaEventSynchronize(end));
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, end));
    timings.push_back(milliseconds);
  }

  unsigned long long* errors = nullptr;
  CUDA_CHECK(cudaMalloc(&errors, sizeof(*errors)));
  CUDA_CHECK(cudaMemsetAsync(errors, 0, sizeof(*errors), stream));
  validate_tiles<<<fill_blocks, fill_threads, 0, stream>>>(data, count, errors);
  unsigned long long host_errors = 0;
  CUDA_CHECK(cudaMemcpyAsync(
    &host_errors, errors, sizeof(host_errors), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  if (host_errors != 0)
    throw std::runtime_error("warp benchmark validation failed");

  std::sort(timings.begin(), timings.end());
  const double best = timings.front();
  const double median = timings[timings.size() / 2];
  std::printf(
    "CCCL CUB WarpMergeSort u32 32x4: items=%llu, best=%.3f ms (%.3f Gitem/s), median=%.3f ms\n",
    static_cast<unsigned long long>(count),
    best,
    static_cast<double>(count) / best / 1.0e6,
    median);

  CUDA_CHECK(cudaFree(errors));
  CUDA_CHECK(cudaEventDestroy(end));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(data));
  return 0;
}
catch (const std::exception& error)
{
  std::fprintf(stderr, "warp merge-sort reference benchmark failed: %s\n", error.what());
  return 1;
}
