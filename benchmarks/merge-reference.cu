#include <cub/device/device_merge.cuh>
#include <thrust/execution_policy.h>
#include <thrust/merge.h>

#include <cuda_runtime.h>

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

__global__ void fill_sorted(std::uint32_t* output, std::int64_t count, std::uint32_t parity)
{
  const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count)
    output[index] = static_cast<std::uint32_t>(index) * 2 + parity;
}

__global__ void fill_sorted_values(std::uint64_t* output, std::int64_t count, std::uint64_t parity)
{
  const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count)
    output[index] = static_cast<std::uint64_t>(index) * 2 + parity;
}

__global__ void validate_iota(const std::uint32_t* input, std::int64_t count, unsigned long long* errors)
{
  const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count && input[index] != static_cast<std::uint32_t>(index))
    atomicAdd(errors, 1ull);
}

__global__ void validate_iota_u64(const std::uint64_t* input, std::int64_t count, unsigned long long* errors)
{
  const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count && input[index] != static_cast<std::uint64_t>(index))
    atomicAdd(errors, 1ull);
}

double elapsed(cudaEvent_t start, cudaEvent_t end)
{
  float milliseconds = 0;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, end));
  return milliseconds;
}

int main(int argc, char** argv)
try
{
  const std::int64_t count = argc > 1 ? std::strtoll(argv[1], nullptr, 10) : (1ll << 24);
  const int iterations = argc > 2 ? std::atoi(argv[2]) : 20;
  const std::int64_t total = count * 2;
  if (count < 0 || total > UINT32_MAX)
    throw std::runtime_error("count must be nonnegative and 2*count <= UINT32_MAX");

  std::uint32_t *first = nullptr, *second = nullptr, *output = nullptr;
  CUDA_CHECK(cudaMalloc(&first, count * sizeof(*first)));
  CUDA_CHECK(cudaMalloc(&second, count * sizeof(*second)));
  CUDA_CHECK(cudaMalloc(&output, total * sizeof(*output)));
  cudaStream_t stream{};
  CUDA_CHECK(cudaStreamCreate(&stream));
  const int threads = 256;
  const int input_blocks = static_cast<int>((count + threads - 1) / threads);
  fill_sorted<<<input_blocks, threads, 0, stream>>>(first, count, 0);
  fill_sorted<<<input_blocks, threads, 0, stream>>>(second, count, 1);
  CUDA_CHECK(cudaGetLastError());

  std::size_t temp_bytes = 0;
  CUDA_CHECK(cub::DeviceMerge::MergeKeys(
    nullptr, temp_bytes, first, count, second, count, output, cuda::std::less<>{}, stream));
  void* temp = nullptr;
  CUDA_CHECK(cudaMalloc(&temp, temp_bytes));

  for (int i = 0; i < 3; ++i)
    CUDA_CHECK(cub::DeviceMerge::MergeKeys(
      temp, temp_bytes, first, count, second, count, output, cuda::std::less<>{}, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaEvent_t start{}, end{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));
  std::vector<double> cub_times;
  for (int i = 0; i < iterations; ++i)
  {
    CUDA_CHECK(cudaEventRecord(start, stream));
    CUDA_CHECK(cub::DeviceMerge::MergeKeys(
      temp, temp_bytes, first, count, second, count, output, cuda::std::less<>{}, stream));
    CUDA_CHECK(cudaEventRecord(end, stream));
    CUDA_CHECK(cudaEventSynchronize(end));
    cub_times.push_back(elapsed(start, end));
  }

  std::vector<double> thrust_times;
  for (int i = 0; i < iterations; ++i)
  {
    CUDA_CHECK(cudaEventRecord(start, stream));
    thrust::merge(
      thrust::cuda::par.on(stream), first, first + count, second, second + count, output);
    CUDA_CHECK(cudaEventRecord(end, stream));
    CUDA_CHECK(cudaEventSynchronize(end));
    thrust_times.push_back(elapsed(start, end));
  }

  unsigned long long* errors = nullptr;
  CUDA_CHECK(cudaMalloc(&errors, sizeof(*errors)));
  CUDA_CHECK(cudaMemsetAsync(errors, 0, sizeof(*errors), stream));
  const int output_blocks = static_cast<int>((total + threads - 1) / threads);
  validate_iota<<<output_blocks, threads, 0, stream>>>(output, total, errors);
  unsigned long long host_errors = 0;
  CUDA_CHECK(cudaMemcpyAsync(
    &host_errors, errors, sizeof(host_errors), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  if (host_errors != 0)
    throw std::runtime_error("device validation failed");

  std::sort(cub_times.begin(), cub_times.end());
  std::sort(thrust_times.begin(), thrust_times.end());
  const auto print = [&](const char* name, const std::vector<double>& times) {
    std::printf(
      "%s u32: inputs=%lld+%lld, best=%.3f ms (%.3f Gitem/s), median=%.3f ms, temp=%zu bytes\n",
      name,
      static_cast<long long>(count),
      static_cast<long long>(count),
      times.front(),
      static_cast<double>(total) / times.front() / 1.0e6,
      times[times.size() / 2],
      temp_bytes);
  };
  print("CCCL CUB DeviceMerge", cub_times);
  print("CCCL Thrust merge", thrust_times);

  std::uint64_t *first_values = nullptr, *second_values = nullptr, *output_values = nullptr;
  CUDA_CHECK(cudaMalloc(&first_values, count * sizeof(*first_values)));
  CUDA_CHECK(cudaMalloc(&second_values, count * sizeof(*second_values)));
  CUDA_CHECK(cudaMalloc(&output_values, total * sizeof(*output_values)));
  fill_sorted_values<<<input_blocks, threads, 0, stream>>>(first_values, count, 0);
  fill_sorted_values<<<input_blocks, threads, 0, stream>>>(second_values, count, 1);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaFree(temp));
  temp = nullptr;
  temp_bytes = 0;
  CUDA_CHECK(cub::DeviceMerge::MergePairs(
    nullptr,
    temp_bytes,
    first,
    first_values,
    count,
    second,
    second_values,
    count,
    output,
    output_values,
    cuda::std::less<>{},
    stream));
  CUDA_CHECK(cudaMalloc(&temp, temp_bytes));
  for (int i = 0; i < 3; ++i)
    CUDA_CHECK(cub::DeviceMerge::MergePairs(
      temp,
      temp_bytes,
      first,
      first_values,
      count,
      second,
      second_values,
      count,
      output,
      output_values,
      cuda::std::less<>{},
      stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<double> cub_pair_times;
  for (int i = 0; i < iterations; ++i)
  {
    CUDA_CHECK(cudaEventRecord(start, stream));
    CUDA_CHECK(cub::DeviceMerge::MergePairs(
      temp,
      temp_bytes,
      first,
      first_values,
      count,
      second,
      second_values,
      count,
      output,
      output_values,
      cuda::std::less<>{},
      stream));
    CUDA_CHECK(cudaEventRecord(end, stream));
    CUDA_CHECK(cudaEventSynchronize(end));
    cub_pair_times.push_back(elapsed(start, end));
  }
  std::vector<double> thrust_pair_times;
  for (int i = 0; i < iterations; ++i)
  {
    CUDA_CHECK(cudaEventRecord(start, stream));
    thrust::merge_by_key(
      thrust::cuda::par.on(stream),
      first,
      first + count,
      second,
      second + count,
      first_values,
      second_values,
      output,
      output_values);
    CUDA_CHECK(cudaEventRecord(end, stream));
    CUDA_CHECK(cudaEventSynchronize(end));
    thrust_pair_times.push_back(elapsed(start, end));
  }
  CUDA_CHECK(cudaMemsetAsync(errors, 0, sizeof(*errors), stream));
  validate_iota_u64<<<output_blocks, threads, 0, stream>>>(output_values, total, errors);
  CUDA_CHECK(cudaMemcpyAsync(
    &host_errors, errors, sizeof(host_errors), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  if (host_errors != 0)
    throw std::runtime_error("device pair validation failed");
  std::sort(cub_pair_times.begin(), cub_pair_times.end());
  std::sort(thrust_pair_times.begin(), thrust_pair_times.end());
  const auto print_pairs = [&](const char* name, const std::vector<double>& times) {
    std::printf(
      "%s u32/u64 pairs: inputs=%lld+%lld, best=%.3f ms (%.3f Gpair/s), median=%.3f ms, temp=%zu bytes\n",
      name,
      static_cast<long long>(count),
      static_cast<long long>(count),
      times.front(),
      static_cast<double>(total) / times.front() / 1.0e6,
      times[times.size() / 2],
      temp_bytes);
  };
  print_pairs("CCCL CUB DeviceMerge", cub_pair_times);
  print_pairs("CCCL Thrust merge_by_key", thrust_pair_times);

  CUDA_CHECK(cudaFree(errors));
  CUDA_CHECK(cudaFree(temp));
  CUDA_CHECK(cudaFree(output_values));
  CUDA_CHECK(cudaFree(second_values));
  CUDA_CHECK(cudaFree(first_values));
  CUDA_CHECK(cudaFree(output));
  CUDA_CHECK(cudaFree(second));
  CUDA_CHECK(cudaFree(first));
  CUDA_CHECK(cudaEventDestroy(end));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
catch (const std::exception& error)
{
  std::fprintf(stderr, "merge reference benchmark failed: %s\n", error.what());
  return 1;
}
