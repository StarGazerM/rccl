#define CCCL_IGNORE_DEPRECATED_API

#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/tabulate.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#define CUDA_CHECK(call) do { auto status = (call); if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status)); } while (false)

namespace {
constexpr std::size_t items = 16 * 1024 * 1024;
constexpr int warmups = 10;
constexpr int samples = 31;
constexpr std::uint32_t mask = 0x5a5a1234U;

struct input_value {
  __device__ std::uint32_t operator()(std::size_t index) const {
    return static_cast<std::uint32_t>(index) ^ mask;
  }
};

struct permutation_value {
  __device__ std::uint32_t operator()(std::size_t index) const {
    return static_cast<std::uint32_t>(index) * 65537U & (static_cast<std::uint32_t>(items) - 1U);
  }
};

template <class Submit>
std::pair<float, float> timings(cudaStream_t stream, Submit submit) {
  for (int iteration = 0; iteration < warmups; ++iteration) submit();
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaEvent_t start{}, end{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));
  std::vector<float> values;
  values.reserve(samples);
  for (int sample = 0; sample < samples; ++sample) {
    CUDA_CHECK(cudaEventRecord(start, stream));
    submit();
    CUDA_CHECK(cudaEventRecord(end, stream));
    CUDA_CHECK(cudaEventSynchronize(end));
    float elapsed = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, end));
    values.push_back(elapsed);
  }
  CUDA_CHECK(cudaEventDestroy(end));
  CUDA_CHECK(cudaEventDestroy(start));
  std::sort(values.begin(), values.end());
  return {values.front(), values[values.size() / 2]};
}

void check_endpoints(
    std::uint32_t const* output,
    std::uint32_t expected_first,
    std::uint32_t expected_last,
    char const* algorithm) {
  std::uint32_t first{}, last{};
  CUDA_CHECK(cudaMemcpy(&first, output, sizeof(first), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&last, output + items - 1, sizeof(last), cudaMemcpyDeviceToHost));
  if (first != expected_first || last != expected_last) {
    throw std::runtime_error(std::string(algorithm) + " result mismatch");
  }
}

void report(char const* algorithm, std::pair<float, float> result) {
  std::printf(
      "implementation=cccl algorithm=%s type=u32 items=%zu best_ms=%.6f median_ms=%.6f\n",
      algorithm, items, result.first, result.second);
}
}  // namespace

int main() try {
  cudaStream_t stream{};
  CUDA_CHECK(cudaStreamCreate(&stream));
  std::uint32_t* input = nullptr;
  std::uint32_t* indices = nullptr;
  std::uint32_t* output = nullptr;
  CUDA_CHECK(cudaMalloc(&input, items * sizeof(*input)));
  CUDA_CHECK(cudaMalloc(&indices, items * sizeof(*indices)));
  CUDA_CHECK(cudaMalloc(&output, items * sizeof(*output)));
  auto policy = thrust::cuda::par_nosync.on(stream);

  thrust::tabulate(policy, input, input + items, input_value{});
  thrust::tabulate(policy, indices, indices + items, permutation_value{});
  CUDA_CHECK(cudaStreamSynchronize(stream));

  auto copy = timings(stream, [&] { thrust::copy_n(policy, input, items, output); });
  check_endpoints(output, mask, static_cast<std::uint32_t>(items - 1) ^ mask, "copy");

  auto gather = timings(stream, [&] { thrust::gather(policy, indices, indices + items, input, output); });
  auto last_index =
      static_cast<std::uint32_t>(items - 1) * 65537U & (static_cast<std::uint32_t>(items) - 1U);
  check_endpoints(output, mask, last_index ^ mask, "gather");

  auto fill = timings(stream, [&] { thrust::fill_n(policy, output, items, 0xdeadbeefU); });
  check_endpoints(output, 0xdeadbeefU, 0xdeadbeefU, "fill");

  auto sequence = timings(stream, [&] { thrust::sequence(policy, output, output + items, 7U, 3U); });
  check_endpoints(
      output, 7U, 7U + 3U * static_cast<std::uint32_t>(items - 1), "sequence");

  report("copy", copy);
  report("gather", gather);
  report("fill", fill);
  report("sequence", sequence);

  CUDA_CHECK(cudaFree(output));
  CUDA_CHECK(cudaFree(indices));
  CUDA_CHECK(cudaFree(input));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
} catch (std::exception const& error) {
  std::fprintf(stderr, "Thrust copy/gather/generation reference failed: %s\n", error.what());
  return 1;
}
