#include <cub/warp/warp_reduce.cuh>

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
constexpr int warp_threads  = 32;
constexpr int blocks         = 32768;
constexpr int reductions_per_value = 4;
constexpr int warmups        = 20;
constexpr int samples        = 21;
constexpr int launches       = 100;
constexpr std::size_t items  = static_cast<std::size_t>(block_threads) * blocks;
constexpr std::size_t warps  = items / warp_threads;

__global__ __launch_bounds__(block_threads) void reduce_warps(std::int32_t* values)
{
  using reduce_t = cub::WarpReduce<std::int32_t, warp_threads>;
  __shared__ typename reduce_t::TempStorage storage[block_threads / warp_threads];

  std::size_t const index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int const warp          = threadIdx.x / warp_threads;
  reduce_t reduce(storage[warp]);
  std::int32_t const value  = values[index];
  std::int32_t const first  = reduce.Sum(value);
  std::int32_t const second = reduce.Sum(value ^ 0x55);
  std::int32_t const third  = reduce.Sum(value * 3);
  std::int32_t const fourth = reduce.Sum(value - 7);
  if ((threadIdx.x % warp_threads) == 0)
    values[index] = first + second + third + fourth;
}

std::int32_t input_value(std::size_t index)
{
  return static_cast<std::int32_t>((index * 17 + 3) % 251) - 125;
}
} // namespace

int main()
try
{
  std::vector<std::int32_t> host_input(items);
  for (std::size_t index = 0; index < items; ++index)
    host_input[index] = input_value(index);

  std::int32_t* values = nullptr;
  CUDA_CHECK(cudaMalloc(&values, items * sizeof(*values)));
  CUDA_CHECK(cudaMemcpy(values, host_input.data(), items * sizeof(*values), cudaMemcpyHostToDevice));

  for (int iteration = 0; iteration < warmups; ++iteration)
    reduce_warps<<<blocks, block_threads>>>(values);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start{}, end{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));
  std::vector<float> timings;
  timings.reserve(samples);
  for (int sample = 0; sample < samples; ++sample)
  {
    CUDA_CHECK(cudaEventRecord(start));
    for (int iteration = 0; iteration < launches; ++iteration)
      reduce_warps<<<blocks, block_threads>>>(values);
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    float milliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, end));
    timings.push_back(milliseconds / launches);
  }

  CUDA_CHECK(cudaMemcpy(values, host_input.data(), items * sizeof(*values), cudaMemcpyHostToDevice));
  reduce_warps<<<blocks, block_threads>>>(values);
  std::vector<std::int32_t> host_output(items);
  CUDA_CHECK(cudaMemcpy(host_output.data(), values, items * sizeof(*values), cudaMemcpyDeviceToHost));
  for (std::size_t warp = 0; warp < warps; ++warp)
  {
    std::int32_t expected = 0;
    for (std::size_t lane = 0; lane < warp_threads; ++lane)
    {
      std::int32_t const value = input_value(warp * warp_threads + lane);
      expected += value + (value ^ 0x55) + value * 3 + (value - 7);
    }
    if (host_output[warp * warp_threads] != expected)
      throw std::runtime_error("CUB WarpReduce result mismatch");
  }

  std::sort(timings.begin(), timings.end());
  std::printf(
    "implementation=cccl primitive=warp_reduce type=i32 width=32 reductions_per_value=%d items=%zu warps=%zu best_ms=%.6f median_ms=%.6f\n",
    reductions_per_value,
    items,
    warps,
    timings.front(),
    timings[timings.size() / 2]);

  CUDA_CHECK(cudaEventDestroy(end));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaFree(values));
  return 0;
}
catch (std::exception const& error)
{
  std::fprintf(stderr, "CCCL WarpReduce reference failed: %s\n", error.what());
  return 1;
}
