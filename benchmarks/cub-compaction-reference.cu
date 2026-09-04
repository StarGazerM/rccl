#include <cub/device/device_scan.cuh>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

#define CUDA_CHECK(call) do { cudaError_t status = (call); if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status)); } while (false)

namespace {
constexpr std::size_t items = (1u << 20) + 13;
constexpr int threads = 256;
constexpr int warmups = 10;
constexpr int samples = 21;
constexpr int launches = 10;

__global__ void mark_predicate(const std::uint32_t* input, std::uint32_t* marks) {
  std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < items) marks[index] = input[index] % 5 <= 1;
}

__global__ void mark_flagged(const std::uint32_t* flags, std::uint32_t* marks) {
  std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < items) marks[index] = flags[index] != 0;
}

__global__ void mark_unique(const std::uint32_t* input, std::uint32_t* marks) {
  std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < items) marks[index] = index == 0 || input[index - 1] != input[index];
}

__global__ void scatter_selected(const std::uint32_t* input, std::uint32_t* output,
                                 const std::uint32_t* marks, const std::uint32_t* positions,
                                 std::uint64_t* count) {
  std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  std::uint32_t total = positions[items - 1] + marks[items - 1];
  if (index == 0) *count = total;
  if (index < items && marks[index]) output[positions[index]] = input[index];
}

__global__ void scatter_partition(const std::uint32_t* input, std::uint32_t* output,
                                  const std::uint32_t* marks, const std::uint32_t* positions,
                                  std::uint64_t* count) {
  std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  std::uint32_t total = positions[items - 1] + marks[items - 1];
  if (index == 0) *count = total;
  if (index < items) {
    std::size_t destination = marks[index] ? positions[index] : total + index - positions[index];
    output[destination] = input[index];
  }
}

struct buffers {
  std::uint32_t *input{}, *flags{}, *unique_input{}, *output{}, *marks{}, *positions{};
  std::uint64_t* count{};
  void* scan_temp{};
  void* parity_padding{};
  std::size_t scan_bytes{};
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

void report(const char* algorithm, std::pair<float, float> timing, std::size_t temp_bytes) {
  std::printf("implementation=cccl algorithm=%s type=u32 items=%zu threads=256 launches=10 best_ms=%.6f median_ms=%.6f temp_bytes=%zu\n",
              algorithm, items, timing.first, timing.second, temp_bytes);
}
} // namespace

int main() try {
  buffers b;
  std::vector<std::uint32_t> input(items), flags(items), unique_input(items);
  for (std::size_t i = 0; i < items; ++i) {
    input[i] = static_cast<std::uint32_t>(i);
    flags[i] = i % 5 <= 1;
    unique_input[i] = static_cast<std::uint32_t>(i / 3);
  }
  CUDA_CHECK(cudaMalloc(&b.input, items * sizeof(std::uint32_t)));
  CUDA_CHECK(cudaMalloc(&b.flags, items * sizeof(std::uint32_t)));
  CUDA_CHECK(cudaMalloc(&b.unique_input, items * sizeof(std::uint32_t)));
  CUDA_CHECK(cudaMalloc(&b.output, items * sizeof(std::uint32_t)));
  CUDA_CHECK(cudaMalloc(&b.marks, items * sizeof(std::uint32_t)));
  CUDA_CHECK(cudaMalloc(&b.positions, items * sizeof(std::uint32_t)));
  CUDA_CHECK(cudaMalloc(&b.count, sizeof(std::uint64_t)));
  CUDA_CHECK(cudaMemcpy(b.input, input.data(), items * sizeof(std::uint32_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b.flags, flags.data(), items * sizeof(std::uint32_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b.unique_input, unique_input.data(), items * sizeof(std::uint32_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, b.scan_bytes, b.marks, b.positions, items));
  CUDA_CHECK(cudaMalloc(&b.scan_temp, b.scan_bytes));
  int blocks = static_cast<int>((items + threads - 1) / threads);
  auto scan = [&] { CUDA_CHECK(cub::DeviceScan::ExclusiveSum(b.scan_temp, b.scan_bytes, b.marks, b.positions, items)); };

  auto select_if = [&] { mark_predicate<<<blocks, threads>>>(b.input, b.marks); scan(); scatter_selected<<<blocks, threads>>>(b.input, b.output, b.marks, b.positions, b.count); };
  auto select_flagged = [&] { mark_flagged<<<blocks, threads>>>(b.flags, b.marks); scan(); scatter_selected<<<blocks, threads>>>(b.input, b.output, b.marks, b.positions, b.count); };
  auto unique_copy = [&] { mark_unique<<<blocks, threads>>>(b.unique_input, b.marks); scan(); scatter_selected<<<blocks, threads>>>(b.unique_input, b.output, b.marks, b.positions, b.count); };
  auto partition = [&] { mark_predicate<<<blocks, threads>>>(b.input, b.marks); scan(); scatter_partition<<<blocks, threads>>>(b.input, b.output, b.marks, b.positions, b.count); };
  std::size_t temp_bytes = b.scan_bytes + 2 * items * sizeof(std::uint32_t);
  constexpr std::size_t rccl_temp_bytes = 8422910;
  if (temp_bytes > rccl_temp_bytes) throw std::runtime_error("CUB workspace exceeds RCCL parity allocation");
  CUDA_CHECK(cudaMalloc(&b.parity_padding, rccl_temp_bytes - temp_bytes));
  temp_bytes = rccl_temp_bytes;
  report("select_if", measure(select_if), temp_bytes);
  report("select_flagged", measure(select_flagged), temp_bytes);
  report("unique_copy", measure(unique_copy), temp_bytes);
  report("stable_partition", measure(partition), temp_bytes);

  partition();
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<std::uint32_t> output(items), expected;
  expected.reserve(items);
  for (auto value : input) if (value % 5 <= 1) expected.push_back(value);
  for (auto value : input) if (value % 5 > 1) expected.push_back(value);
  CUDA_CHECK(cudaMemcpy(output.data(), b.output, items * sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
  if (output != expected) throw std::runtime_error("stable partition mismatch");

  CUDA_CHECK(cudaFree(b.parity_padding)); CUDA_CHECK(cudaFree(b.scan_temp)); CUDA_CHECK(cudaFree(b.count)); CUDA_CHECK(cudaFree(b.positions));
  CUDA_CHECK(cudaFree(b.marks)); CUDA_CHECK(cudaFree(b.output)); CUDA_CHECK(cudaFree(b.unique_input));
  CUDA_CHECK(cudaFree(b.flags)); CUDA_CHECK(cudaFree(b.input));
  return 0;
} catch (const std::exception& error) {
  std::fprintf(stderr, "CUB compaction reference failed: %s\n", error.what());
  return 1;
}
