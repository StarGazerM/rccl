#include <cub/device/device_reduce.cuh>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <utility>
#include <vector>

#define CUDA_CHECK(call) do { cudaError_t status = (call); if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status)); } while (false)

namespace {
constexpr std::size_t items = (1u << 20) + 13;
constexpr int threads = 256;
constexpr int warmups = 10;
constexpr int samples = 21;
constexpr int launches = 10;
constexpr std::size_t rccl_temp_bytes = 8422910;

struct wrapping_add {
  __host__ __device__ std::uint64_t operator()(std::uint64_t left, std::uint64_t right) const {
    return left + right;
  }
};

template <class Operation>
std::pair<float, float> measure(Operation operation) {
  for (int warmup = 0; warmup < warmups; ++warmup)
    for (int launch = 0; launch < launches; ++launch) operation();
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start{}, end{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));
  std::vector<float> timings;
  for (int sample = 0; sample < samples; ++sample) {
    CUDA_CHECK(cudaEventRecord(start));
    for (int launch = 0; launch < launches; ++launch) operation();
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
    timings.push_back(ms / launches);
  }
  CUDA_CHECK(cudaEventDestroy(end));
  CUDA_CHECK(cudaEventDestroy(start));
  std::sort(timings.begin(), timings.end());
  return {timings.front(), timings[timings.size() / 2]};
}
} // namespace

int main() try {
  std::vector<std::uint32_t> keys(items);
  std::vector<std::uint64_t> values(items);
  for (std::size_t index = 0; index < items; ++index) {
    keys[index] = static_cast<std::uint32_t>(index / 3);
    values[index] = static_cast<std::uint64_t>(index % 17 + 1);
  }

  std::uint32_t *d_keys{}, *d_output_keys{};
  std::uint64_t *d_values{}, *d_output_values{}, *d_count{};
  void *d_temp{}, *parity_padding{};
  CUDA_CHECK(cudaMalloc(&d_keys, items * sizeof(*d_keys)));
  CUDA_CHECK(cudaMalloc(&d_values, items * sizeof(*d_values)));
  CUDA_CHECK(cudaMalloc(&d_output_keys, items * sizeof(*d_output_keys)));
  CUDA_CHECK(cudaMalloc(&d_output_values, items * sizeof(*d_output_values)));
  CUDA_CHECK(cudaMalloc(&d_count, sizeof(*d_count)));
  CUDA_CHECK(cudaMemcpy(d_keys, keys.data(), items * sizeof(*d_keys), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_values, values.data(), items * sizeof(*d_values), cudaMemcpyHostToDevice));

  std::size_t temp_bytes = 0;
  CUDA_CHECK(cub::DeviceReduce::ReduceByKey(
      nullptr, temp_bytes, d_keys, d_output_keys, d_values, d_output_values,
      d_count, wrapping_add{}, items));
  if (temp_bytes > rccl_temp_bytes)
    throw std::runtime_error("CUB workspace exceeds RCCL parity allocation");
  CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
  if (rccl_temp_bytes != temp_bytes)
    CUDA_CHECK(cudaMalloc(&parity_padding, rccl_temp_bytes - temp_bytes));

  auto reduce_by_key = [&] {
    CUDA_CHECK(cub::DeviceReduce::ReduceByKey(
        d_temp, temp_bytes, d_keys, d_output_keys, d_values, d_output_values,
        d_count, wrapping_add{}, items));
  };
  auto timing = measure(reduce_by_key);
  std::printf(
      "implementation=cccl algorithm=reduce_by_key type=u32-u64 items=%zu threads=%d launches=%d best_ms=%.6f median_ms=%.6f temp_bytes=%zu\n",
      items, threads, launches, timing.first, timing.second, rccl_temp_bytes);

  reduce_by_key();
  CUDA_CHECK(cudaDeviceSynchronize());
  std::uint64_t groups = 0;
  CUDA_CHECK(cudaMemcpy(&groups, d_count, sizeof(groups), cudaMemcpyDeviceToHost));
  if (groups != (items + 2) / 3) throw std::runtime_error("group count mismatch");
  std::vector<std::uint32_t> output_keys(groups);
  std::vector<std::uint64_t> output_values(groups);
  CUDA_CHECK(cudaMemcpy(output_keys.data(), d_output_keys, groups * sizeof(*d_output_keys), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(output_values.data(), d_output_values, groups * sizeof(*d_output_values), cudaMemcpyDeviceToHost));
  for (std::size_t group = 0; group < groups; ++group) {
    const std::size_t begin = group * 3;
    const std::size_t end = std::min(begin + 3, items);
    std::uint64_t expected = 0;
    for (std::size_t index = begin; index < end; ++index) expected += values[index];
    if (output_keys[group] != group || output_values[group] != expected)
      throw std::runtime_error("reduce-by-key output mismatch");
  }

  if (parity_padding) CUDA_CHECK(cudaFree(parity_padding));
  CUDA_CHECK(cudaFree(d_temp));
  CUDA_CHECK(cudaFree(d_count));
  CUDA_CHECK(cudaFree(d_output_values));
  CUDA_CHECK(cudaFree(d_output_keys));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_keys));
  return 0;
} catch (const std::exception& error) {
  std::fprintf(stderr, "CUB grouped reduction reference failed: %s\n", error.what());
  return 1;
}
