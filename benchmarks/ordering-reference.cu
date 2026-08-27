#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#define CUDA_CHECK(call)                                                                    \
  do                                                                                        \
  {                                                                                         \
    const cudaError_t status = (call);                                                      \
    if (status != cudaSuccess)                                                              \
      throw std::runtime_error(cudaGetErrorString(status));                                 \
  } while (false)

__global__ void fill_random(std::uint32_t* output, std::int64_t count)
{
  const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  std::uint64_t value = static_cast<std::uint64_t>(index) ^ 0x534d4152545f534full;
  value ^= value << 13;
  value ^= value >> 7;
  value ^= value << 17;
  output[index] = static_cast<std::uint32_t>(value);
}

__global__ void validate_order(const std::uint32_t* input, std::int64_t count, unsigned long long* errors)
{
  const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index + 1 < count && input[index] > input[index + 1])
    atomicAdd(errors, 1ull);
}

double elapsed(cudaEvent_t start, cudaEvent_t end)
{
  float milliseconds = 0;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, end));
  return milliseconds;
}

template <class Operation>
std::vector<double> measure(cudaStream_t stream, cudaEvent_t start, cudaEvent_t end, int iterations, Operation operation)
{
  for (int i = 0; i < 3; ++i)
    CUDA_CHECK(operation());
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<double> samples;
  samples.reserve(iterations);
  for (int i = 0; i < iterations; ++i)
  {
    CUDA_CHECK(cudaEventRecord(start, stream));
    CUDA_CHECK(operation());
    CUDA_CHECK(cudaEventRecord(end, stream));
    CUDA_CHECK(cudaEventSynchronize(end));
    samples.push_back(elapsed(start, end));
  }
  std::sort(samples.begin(), samples.end());
  return samples;
}

int main(int argc, char** argv)
try
{
  const std::int64_t count = argc > 1 ? std::strtoll(argv[1], nullptr, 10) : 1'000'003;
  const int iterations = argc > 2 ? std::atoi(argv[2]) : 7;
  if (count <= 0 || iterations <= 0)
    throw std::runtime_error("count and iterations must be positive");

  cudaStream_t stream{};
  CUDA_CHECK(cudaStreamCreate(&stream));
  std::uint32_t *input = nullptr, *output = nullptr, *thrust_keys = nullptr;
  std::uint32_t *thrust_pair_keys = nullptr, *thrust_pair_values = nullptr;
  std::uint32_t *input_values = nullptr, *output_values = nullptr;
  CUDA_CHECK(cudaMalloc(&input, count * sizeof(*input)));
  CUDA_CHECK(cudaMalloc(&output, count * sizeof(*output)));
  CUDA_CHECK(cudaMalloc(&thrust_keys, count * sizeof(*thrust_keys)));
  CUDA_CHECK(cudaMalloc(&thrust_pair_keys, count * sizeof(*thrust_pair_keys)));
  CUDA_CHECK(cudaMalloc(&thrust_pair_values, count * sizeof(*thrust_pair_values)));
  CUDA_CHECK(cudaMalloc(&input_values, count * sizeof(*input_values)));
  CUDA_CHECK(cudaMalloc(&output_values, count * sizeof(*output_values)));
  constexpr int threads = 256;
  const int blocks = static_cast<int>((count + threads - 1) / threads);
  fill_random<<<blocks, threads, 0, stream>>>(input, count);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpyAsync(
    input_values, input, count * sizeof(*input), cudaMemcpyDeviceToDevice, stream));
  std::vector<std::uint32_t> expected(static_cast<std::size_t>(count));
  for (std::int64_t index = 0; index < count; ++index)
  {
    std::uint64_t value = static_cast<std::uint64_t>(index) ^ 0x534d4152545f534full;
    value ^= value << 13;
    value ^= value >> 7;
    value ^= value << 17;
    expected[static_cast<std::size_t>(index)] = static_cast<std::uint32_t>(value);
  }
  std::sort(expected.begin(), expected.end());

  std::size_t radix_temp_bytes = 0;
  CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
    nullptr, radix_temp_bytes, input, output, count, 0, 32, stream));
  std::size_t radix_pairs_temp_bytes = 0;
  CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
    nullptr,
    radix_pairs_temp_bytes,
    input,
    output,
    input_values,
    output_values,
    count,
    0,
    32,
    stream));
  std::size_t merge_temp_bytes = 0;
  CUDA_CHECK(cub::DeviceMergeSort::SortKeysCopy(
    nullptr, merge_temp_bytes, input, output, count, cuda::std::less<>{}, stream));
  std::size_t merge_pairs_temp_bytes = 0;
  CUDA_CHECK(cub::DeviceMergeSort::SortPairsCopy(
    nullptr,
    merge_pairs_temp_bytes,
    input,
    input_values,
    output,
    output_values,
    count,
    cuda::std::less<>{},
    stream));
  const std::size_t temp_bytes =
    std::max({radix_temp_bytes, radix_pairs_temp_bytes, merge_temp_bytes, merge_pairs_temp_bytes});
  void* temp = nullptr;
  CUDA_CHECK(cudaMalloc(&temp, temp_bytes));

  cudaEvent_t start{}, end{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));
  const auto radix = measure(stream, start, end, iterations, [&] {
    return cub::DeviceRadixSort::SortKeys(
      temp, radix_temp_bytes, input, output, count, 0, 32, stream);
  });
  std::vector<std::uint32_t> radix_output(static_cast<std::size_t>(count));
  CUDA_CHECK(cudaMemcpy(
    radix_output.data(), output, count * sizeof(*output), cudaMemcpyDeviceToHost));
  if (radix_output != expected)
    throw std::runtime_error("CUB radix exact-output validation failed");
  const auto radix_pairs = measure(stream, start, end, iterations, [&] {
    return cub::DeviceRadixSort::SortPairs(
      temp,
      radix_pairs_temp_bytes,
      input,
      output,
      input_values,
      output_values,
      count,
      0,
      32,
      stream);
  });
  std::vector<std::uint32_t> radix_pair_keys(static_cast<std::size_t>(count));
  std::vector<std::uint32_t> radix_pair_values(static_cast<std::size_t>(count));
  CUDA_CHECK(cudaMemcpy(
    radix_pair_keys.data(), output, count * sizeof(*output), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(
    radix_pair_values.data(), output_values, count * sizeof(*output_values), cudaMemcpyDeviceToHost));
  if (radix_pair_keys != expected || radix_pair_values != expected)
    throw std::runtime_error("CUB radix pair exact-output validation failed");
  const auto merge = measure(stream, start, end, iterations, [&] {
    return cub::DeviceMergeSort::SortKeysCopy(
      temp, merge_temp_bytes, input, output, count, cuda::std::less<>{}, stream);
  });
  std::vector<std::uint32_t> merge_output(static_cast<std::size_t>(count));
  CUDA_CHECK(cudaMemcpy(
    merge_output.data(), output, count * sizeof(*output), cudaMemcpyDeviceToHost));
  if (merge_output != expected)
    throw std::runtime_error("CUB merge-sort exact-output validation failed");
  const auto merge_pairs = measure(stream, start, end, iterations, [&] {
    return cub::DeviceMergeSort::SortPairsCopy(
      temp,
      merge_pairs_temp_bytes,
      input,
      input_values,
      output,
      output_values,
      count,
      cuda::std::less<>{},
      stream);
  });
  std::vector<std::uint32_t> merge_pair_keys(static_cast<std::size_t>(count));
  std::vector<std::uint32_t> merge_pair_values(static_cast<std::size_t>(count));
  CUDA_CHECK(cudaMemcpy(
    merge_pair_keys.data(), output, count * sizeof(*output), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(
    merge_pair_values.data(), output_values, count * sizeof(*output_values), cudaMemcpyDeviceToHost));
  if (merge_pair_keys != expected || merge_pair_values != expected)
    throw std::runtime_error("CUB merge-sort pair exact-output validation failed");

  // Reset outside the measured interval so every Thrust sample receives the
  // same unsorted keys while timing only the operation, like the CUB samples.
  std::vector<double> thrust_samples;
  thrust_samples.reserve(iterations);
  for (int i = 0; i < iterations + 3; ++i)
  {
    CUDA_CHECK(cudaMemcpyAsync(
      thrust_keys, input, count * sizeof(*input), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaEventRecord(start, stream));
    thrust::sort(thrust::cuda::par.on(stream), thrust_keys, thrust_keys + count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(end, stream));
    CUDA_CHECK(cudaEventSynchronize(end));
    if (i >= 3)
      thrust_samples.push_back(elapsed(start, end));
  }
  std::sort(thrust_samples.begin(), thrust_samples.end());

  std::vector<std::uint32_t> thrust_output(static_cast<std::size_t>(count));
  CUDA_CHECK(cudaMemcpy(
    thrust_output.data(), thrust_keys, count * sizeof(*thrust_keys), cudaMemcpyDeviceToHost));
  if (thrust_output != expected)
    throw std::runtime_error("Thrust exact-output validation failed");

  std::vector<double> thrust_pair_samples;
  thrust_pair_samples.reserve(iterations);
  for (int i = 0; i < iterations + 3; ++i)
  {
    CUDA_CHECK(cudaMemcpyAsync(
      thrust_pair_keys, input, count * sizeof(*input), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
      thrust_pair_values, input_values, count * sizeof(*input_values), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaEventRecord(start, stream));
    thrust::sort_by_key(
      thrust::cuda::par.on(stream), thrust_pair_keys, thrust_pair_keys + count, thrust_pair_values);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(end, stream));
    CUDA_CHECK(cudaEventSynchronize(end));
    if (i >= 3)
      thrust_pair_samples.push_back(elapsed(start, end));
  }
  std::sort(thrust_pair_samples.begin(), thrust_pair_samples.end());
  std::vector<std::uint32_t> thrust_pair_output_keys(static_cast<std::size_t>(count));
  std::vector<std::uint32_t> thrust_pair_output_values(static_cast<std::size_t>(count));
  CUDA_CHECK(cudaMemcpy(
    thrust_pair_output_keys.data(), thrust_pair_keys, count * sizeof(*thrust_pair_keys), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(thrust_pair_output_values.data(),
                        thrust_pair_values,
                        count * sizeof(*thrust_pair_values),
                        cudaMemcpyDeviceToHost));
  if (thrust_pair_output_keys != expected || thrust_pair_output_values != expected)
    throw std::runtime_error("Thrust pair exact-output validation failed");

  unsigned long long* errors = nullptr;
  CUDA_CHECK(cudaMalloc(&errors, sizeof(*errors)));
  CUDA_CHECK(cudaMemsetAsync(errors, 0, sizeof(*errors), stream));
  validate_order<<<blocks, threads, 0, stream>>>(thrust_keys, count, errors);
  unsigned long long host_errors = 0;
  CUDA_CHECK(cudaMemcpyAsync(
    &host_errors, errors, sizeof(host_errors), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  if (host_errors != 0)
    throw std::runtime_error("Thrust order validation failed");

  std::printf(
    "CCCL_ORDERING items=%lld cub_radix_gpu_median_ms=%.6f cub_radix_pairs_gpu_median_ms=%.6f "
    "cub_merge_sort_copy_gpu_median_ms=%.6f cub_merge_sort_pairs_copy_gpu_median_ms=%.6f "
    "thrust_sort_gpu_median_ms=%.6f thrust_sort_by_key_gpu_median_ms=%.6f "
    "radix_temp_bytes=%zu radix_pairs_temp_bytes=%zu merge_temp_bytes=%zu merge_pairs_temp_bytes=%zu\n",
    static_cast<long long>(count),
    radix[radix.size() / 2],
    radix_pairs[radix_pairs.size() / 2],
    merge[merge.size() / 2],
    merge_pairs[merge_pairs.size() / 2],
    thrust_samples[thrust_samples.size() / 2],
    thrust_pair_samples[thrust_pair_samples.size() / 2],
    radix_temp_bytes,
    radix_pairs_temp_bytes,
    merge_temp_bytes,
    merge_pairs_temp_bytes);

  CUDA_CHECK(cudaFree(errors));
  CUDA_CHECK(cudaFree(temp));
  CUDA_CHECK(cudaFree(thrust_pair_values));
  CUDA_CHECK(cudaFree(thrust_pair_keys));
  CUDA_CHECK(cudaFree(thrust_keys));
  CUDA_CHECK(cudaFree(output_values));
  CUDA_CHECK(cudaFree(input_values));
  CUDA_CHECK(cudaFree(output));
  CUDA_CHECK(cudaFree(input));
  CUDA_CHECK(cudaEventDestroy(end));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
catch (const std::exception& error)
{
  std::fprintf(stderr, "ordering reference benchmark failed: %s\n", error.what());
  return 1;
}
