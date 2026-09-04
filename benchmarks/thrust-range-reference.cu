#define CCCL_IGNORE_DEPRECATED_API

#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

#define CUDA_CHECK(call) do { auto status = (call); if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status)); } while (false)

namespace {
constexpr std::size_t items = 8 * 1024 * 1024;
constexpr int warmups = 20;
constexpr int samples = 21;
constexpr int launches = 100;
constexpr std::uint32_t mask = 0x5a5a1234U;

struct arithmetic_value {
  std::uint32_t start;
  std::uint32_t step;

  __device__ std::uint32_t operator()(std::uint32_t index) const {
    return start + step * index;
  }
};

struct transform_value {
  __device__ std::uint32_t operator()(std::uint32_t value) const {
    return value * 17U + 3U ^ mask;
  }
};

struct write_value {
  std::uint32_t* output;
  __device__ void operator()(std::uint32_t index) const {
    output[index] = index * 17U + 3U ^ mask;
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
    for (int iteration = 0; iteration < launches; ++iteration) submit();
    CUDA_CHECK(cudaEventRecord(end, stream));
    CUDA_CHECK(cudaEventSynchronize(end));
    float elapsed = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, end));
    values.push_back(elapsed / launches);
  }
  CUDA_CHECK(cudaEventDestroy(end));
  CUDA_CHECK(cudaEventDestroy(start));
  std::sort(values.begin(), values.end());
  return {values.front(), values[values.size() / 2]};
}
}

int main() try {
  cudaStream_t stream{};
  CUDA_CHECK(cudaStreamCreate(&stream));
  std::uint32_t* output = nullptr;
  CUDA_CHECK(cudaMalloc(&output, items * sizeof(*output)));
  auto counting = thrust::make_counting_iterator<std::uint32_t>(0);
  auto begin = thrust::make_transform_iterator(counting, arithmetic_value{0U, 1U});
  auto policy = thrust::cuda::par_nosync.on(stream);

  auto transform = timings(stream, [&] {
    thrust::transform(policy, begin, begin + items, output, transform_value{});
  });
  auto for_each = timings(stream, [&] {
    thrust::for_each_n(policy, begin, items, write_value{output});
  });

  std::uint32_t last{};
  CUDA_CHECK(cudaMemcpy(&last, output + items - 1, sizeof(last), cudaMemcpyDeviceToHost));
  auto expected = (static_cast<std::uint32_t>(items - 1) * 17U + 3U) ^ mask;
  if (last != expected) throw std::runtime_error("range result mismatch");

  std::printf("implementation=cccl algorithm=transform adaptor=arithmetic type=u32 items=%zu best_ms=%.6f median_ms=%.6f\n", items, transform.first, transform.second);
  std::printf("implementation=cccl algorithm=for_each adaptor=arithmetic type=u32 items=%zu best_ms=%.6f median_ms=%.6f\n", items, for_each.first, for_each.second);
  CUDA_CHECK(cudaFree(output));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
} catch (std::exception const& error) {
  std::fprintf(stderr, "Thrust range reference failed: %s\n", error.what());
  return 1;
}
