#include <cub/device/device_merge_sort.cuh>

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
constexpr std::size_t parity_temp_bytes = 25174344;

struct alignas(4) Row4 {
  std::uint32_t fields[4];
  friend bool operator==(const Row4&, const Row4&) = default;
};
static_assert(sizeof(Row4) == 16);

struct ascending {
  __host__ __device__ bool operator()(const Row4& left, const Row4& right) const {
#pragma unroll
    for (int field = 0; field < 4; ++field) {
      if (left.fields[field] < right.fields[field]) return true;
      if (left.fields[field] > right.fields[field]) return false;
    }
    return false;
  }
};

Row4 make_row(std::size_t index) {
  Row4 row{};
  for (std::size_t field = 0; field < 4; ++field)
    row.fields[field] = static_cast<std::uint32_t>((index * (field * 6 + 5) + field * 3) % (11 + field * 2));
  return row;
}

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
  std::vector<Row4> rows(items);
  std::vector<std::uint64_t> payloads(items);
  for (std::size_t index = 0; index < items; ++index) {
    rows[index] = make_row(index);
    payloads[index] = index;
  }
  std::vector<std::pair<Row4, std::uint64_t>> expected(items);
  for (std::size_t index = 0; index < items; ++index) expected[index] = {rows[index], payloads[index]};
  std::stable_sort(expected.begin(), expected.end(), [](const auto& left, const auto& right) {
    return ascending{}(left.first, right.first);
  });

  Row4 *d_source_rows{}, *d_rows{};
  std::uint64_t *d_source_payloads{}, *d_payloads{};
  void *d_temp{}, *parity_padding{};
  CUDA_CHECK(cudaMalloc(&d_source_rows, items * sizeof(*d_source_rows)));
  CUDA_CHECK(cudaMalloc(&d_rows, items * sizeof(*d_rows)));
  CUDA_CHECK(cudaMalloc(&d_source_payloads, items * sizeof(*d_source_payloads)));
  CUDA_CHECK(cudaMalloc(&d_payloads, items * sizeof(*d_payloads)));
  CUDA_CHECK(cudaMemcpy(d_source_rows, rows.data(), items * sizeof(*d_source_rows), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_source_payloads, payloads.data(), items * sizeof(*d_source_payloads), cudaMemcpyHostToDevice));

  std::size_t temp_bytes = 0;
  CUDA_CHECK(cub::DeviceMergeSort::StableSortPairs(
      nullptr, temp_bytes, d_rows, d_payloads, static_cast<int>(items), ascending{}));
  CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
  if (temp_bytes > parity_temp_bytes) throw std::runtime_error("CUB workspace exceeds RCCL parity allocation");
  CUDA_CHECK(cudaMalloc(&parity_padding, parity_temp_bytes - temp_bytes));
  auto sort = [&] {
    CUDA_CHECK(cudaMemcpy(d_rows, d_source_rows, items * sizeof(*d_rows), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_payloads, d_source_payloads, items * sizeof(*d_payloads), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cub::DeviceMergeSort::StableSortPairs(
        d_temp, temp_bytes, d_rows, d_payloads, static_cast<int>(items), ascending{}));
  };
  auto timing = measure(sort);
  std::printf(
      "implementation=cccl algorithm=stable_record_sort type=row4-u64 items=%zu threads=%d launches=%d best_ms=%.6f median_ms=%.6f temp_bytes=%zu\n",
      items, threads, launches, timing.first, timing.second, parity_temp_bytes);

  sort();
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<Row4> actual_rows(items);
  std::vector<std::uint64_t> actual_payloads(items);
  CUDA_CHECK(cudaMemcpy(actual_rows.data(), d_rows, items * sizeof(*d_rows), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(actual_payloads.data(), d_payloads, items * sizeof(*d_payloads), cudaMemcpyDeviceToHost));
  for (std::size_t index = 0; index < items; ++index) {
    if (!(actual_rows[index] == expected[index].first) || actual_payloads[index] != expected[index].second)
      throw std::runtime_error("stable record/payload output mismatch");
  }

  CUDA_CHECK(cudaFree(parity_padding));
  CUDA_CHECK(cudaFree(d_temp));
  CUDA_CHECK(cudaFree(d_payloads));
  CUDA_CHECK(cudaFree(d_source_payloads));
  CUDA_CHECK(cudaFree(d_rows));
  CUDA_CHECK(cudaFree(d_source_rows));
  return 0;
} catch (const std::exception& error) {
  std::fprintf(stderr, "CUB record sort reference failed: %s\n", error.what());
  return 1;
}
