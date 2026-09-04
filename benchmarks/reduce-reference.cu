#include <cub/device/device_reduce.cuh>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>
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
constexpr int warmups  = 20;
constexpr int samples  = 21;
constexpr int launches = 50;

template <typename T>
T one()
{
  return T{1};
}

template <typename T>
const char* type_name();

template <>
const char* type_name<std::uint32_t>()
{
  return "u32";
}

template <>
const char* type_name<std::int32_t>()
{
  return "i32";
}

template <>
const char* type_name<float>()
{
  return "f32";
}

template <typename T>
void check_result(T value, std::size_t items)
{
  if constexpr (std::is_floating_point_v<T>)
  {
    if (value != static_cast<T>(items))
      throw std::runtime_error("DeviceReduce::Sum floating result mismatch");
  }
  else
  {
    if (value != static_cast<T>(items))
      throw std::runtime_error("DeviceReduce::Sum integer result mismatch");
  }
}

template <typename T>
void benchmark(std::size_t items)
{
  std::vector<T> host_input(items, one<T>());
  T* d_in  = nullptr;
  T* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, items * sizeof(T)));
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(T)));
  CUDA_CHECK(cudaMemcpy(d_in, host_input.data(), items * sizeof(T), cudaMemcpyHostToDevice));

  void* d_temp_storage          = nullptr;
  std::size_t temp_storage_size = 0;
  CUDA_CHECK(cub::DeviceReduce::Sum(d_temp_storage, temp_storage_size, d_in, d_out, items));
  CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_size));

  for (int iteration = 0; iteration < warmups; ++iteration)
    CUDA_CHECK(cub::DeviceReduce::Sum(d_temp_storage, temp_storage_size, d_in, d_out, items));
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
      CUDA_CHECK(cub::DeviceReduce::Sum(d_temp_storage, temp_storage_size, d_in, d_out, items));
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    float milliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, end));
    timings.push_back(milliseconds / launches);
  }

  T host_output{};
  CUDA_CHECK(cudaMemcpy(&host_output, d_out, sizeof(T), cudaMemcpyDeviceToHost));
  check_result(host_output, items);

  std::sort(timings.begin(), timings.end());
  std::printf(
    "CCCL_REDUCE algorithm=device_reduce_sum type=%s items=%zu temp_bytes=%zu best_ms=%.9f median_ms=%.9f\n",
    type_name<T>(),
    items,
    temp_storage_size,
    timings.front(),
    timings[timings.size() / 2]);

  CUDA_CHECK(cudaEventDestroy(end));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaFree(d_temp_storage));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_in));
}
} // namespace

int main(int argc, char** argv)
try
{
  std::size_t const items = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 64ull * 1024ull * 1024ull;
  if (items == 0)
    throw std::runtime_error("items must be non-zero");

  benchmark<std::uint32_t>(items);
  benchmark<std::int32_t>(items);
  benchmark<float>(items);
  return 0;
}
catch (std::exception const& error)
{
  std::fprintf(stderr, "CUB DeviceReduce reference benchmark failed: %s\n", error.what());
  return 1;
}
