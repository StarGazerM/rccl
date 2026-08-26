#define CCCL_IGNORE_DEPRECATED_API

#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include <cuda/iterator>
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
constexpr std::size_t items = 8 * 1024 * 1024;
constexpr int warmups        = 20;
constexpr int samples        = 21;
constexpr int launches       = 100;
constexpr std::uint32_t add  = 10;

std::uint32_t input_value(std::size_t index)
{
  return static_cast<std::uint32_t>((index * 17 + 3) % 251);
}
} // namespace

int main()
try
{
  std::vector<std::uint32_t> host_input(items);
  for (std::size_t index = 0; index < items; ++index)
    host_input[index] = input_value(index);

  std::uint32_t* raw = nullptr;
  CUDA_CHECK(cudaMalloc(&raw, items * sizeof(*raw)));
  CUDA_CHECK(cudaMemcpy(raw, host_input.data(), items * sizeof(*raw), cudaMemcpyHostToDevice));
  auto data     = thrust::device_pointer_cast(raw);
  auto repeated = cuda::constant_iterator<std::uint32_t>(add);

  for (int iteration = 0; iteration < warmups; ++iteration)
    thrust::transform(data, data + items, repeated, data, thrust::plus<std::uint32_t>{});
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
      thrust::transform(data, data + items, repeated, data, thrust::plus<std::uint32_t>{});
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    float milliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, end));
    timings.push_back(milliseconds / launches);
  }

  std::vector<std::uint32_t> host_output(items);
  CUDA_CHECK(cudaMemcpy(host_output.data(), raw, items * sizeof(*raw), cudaMemcpyDeviceToHost));
  std::uint32_t const increment = add * (warmups + samples * launches);
  for (std::size_t index = 0; index < items; ++index)
  {
    if (host_output[index] != input_value(index) + increment)
      throw std::runtime_error("Thrust constant-iterator transform result mismatch");
  }

  std::sort(timings.begin(), timings.end());
  std::printf(
    "implementation=cccl algorithm=transform adaptor=constant_iterator type=u32 items=%zu best_ms=%.6f median_ms=%.6f\n",
    items,
    timings.front(),
    timings[timings.size() / 2]);

  CUDA_CHECK(cudaEventDestroy(end));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaFree(raw));
  return 0;
}
catch (std::exception const& error)
{
  std::fprintf(stderr, "CCCL Thrust constant transform reference failed: %s\n", error.what());
  return 1;
}
