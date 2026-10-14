#define CCCL_IGNORE_DEPRECATED_API
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do { auto status = (call); if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status)); } while (false)

namespace {
constexpr int warmups = 10;
constexpr int samples = 31;
constexpr std::size_t seed = 19;
using clock_type = std::chrono::steady_clock;
double elapsed_ms(clock_type::time_point start) {
  return std::chrono::duration<double, std::milli>(clock_type::now() - start).count();
}
struct compare_keys {
  bool descending;
  __host__ __device__ bool operator()(std::uint64_t left, std::uint64_t right) const {
    return descending ? right < left : left < right;
  }
};

template <class Submit>
std::vector<float> timings(cudaStream_t stream, Submit submit) {
  for (int iteration = 0; iteration < warmups; ++iteration) {
    submit();
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  std::vector<float> values;
  values.reserve(samples);
  for (int sample = 0; sample < samples; ++sample) {
    // Match Rust stream.record_event: create each event immediately before recording it.
    cudaEvent_t start{}, end{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventRecord(start, stream));
    submit();
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(end, stream));
    CUDA_CHECK(cudaEventSynchronize(end));
    float elapsed{};
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, end));
    values.push_back(elapsed);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaEventDestroy(start));
  }
  std::sort(values.begin(), values.end());
  return values;
}

void check(std::uint64_t const* device, std::vector<std::uint64_t> const& expected) {
  std::vector<std::uint64_t> actual(expected.size());
  CUDA_CHECK(cudaMemcpy(actual.data(), device, actual.size() * sizeof(actual[0]), cudaMemcpyDeviceToHost));
  if (actual != expected) throw std::runtime_error("batched search differs from CPU oracle");
}
}  // namespace

int main() try {
  auto cold = clock_type::now();
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaFree(nullptr));
  std::printf("setup implementation=cccl context_program_ms=%.6f\n", elapsed_ms(cold));
  cudaStream_t stream{};  // Default stream, matching the Rust benchmark.
  auto policy = thrust::cuda::par_nosync.on(stream);
  for (bool large : {false, true}) {
    std::size_t haystack_len = large ? 1U << 20 : 1024;
    std::size_t needle_len = large ? 1U << 18 : 256;
    char const* size = large ? "large" : "small";
    for (std::string dataset : {"uniform", "duplicates", "misses"}) {
      for (bool descending : {false, true}) {
        auto host_start = clock_type::now();
        std::size_t repeats = dataset == "duplicates" ? 32 : 1;
        std::size_t distinct = haystack_len / repeats;
        std::vector<std::uint64_t> host_haystack(haystack_len), host_needles(needle_len);
        for (std::size_t i = 0; i < haystack_len; ++i) host_haystack[i] = 2 * (i / repeats) + 2;
        if (descending) std::reverse(host_haystack.begin(), host_haystack.end());
        for (std::size_t i = 0; i < needle_len; ++i) {
          host_needles[i] = 2 * ((i * 65537 + seed) % distinct) + 2
              + (dataset == "misses" && i % 8 != 0);
        }
        double host_ms = elapsed_ms(host_start);
        compare_keys compare{descending};
        std::vector<std::uint64_t> expected_begins(needle_len), expected_ends(needle_len);
        for (std::size_t i = 0; i < needle_len; ++i) {
          expected_begins[i] = std::lower_bound(host_haystack.begin(), host_haystack.end(), host_needles[i], compare) - host_haystack.begin();
          expected_ends[i] = std::upper_bound(host_haystack.begin(), host_haystack.end(), host_needles[i], compare) - host_haystack.begin();
        }

        auto setup = clock_type::now();
        std::uint64_t *haystack{}, *needles{}, *begins{}, *ends{};
        CUDA_CHECK(cudaMalloc(&haystack, haystack_len * sizeof(*haystack)));
        CUDA_CHECK(cudaMalloc(&needles, needle_len * sizeof(*needles)));
        CUDA_CHECK(cudaMalloc(&begins, needle_len * sizeof(*begins)));
        CUDA_CHECK(cudaMalloc(&ends, needle_len * sizeof(*ends)));
        CUDA_CHECK(cudaMemcpyAsync(haystack, host_haystack.data(), haystack_len * sizeof(*haystack), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(needles, host_needles.data(), needle_len * sizeof(*needles), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemsetAsync(begins, 0, needle_len * sizeof(*begins), stream));
        CUDA_CHECK(cudaMemsetAsync(ends, 0, needle_len * sizeof(*ends), stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        char const* order = descending ? "descending" : "ascending";
        std::printf("setup implementation=cccl size=%s dataset=%s order=%s host_ms=%.6f allocation_upload_ms=%.6f storage_bytes=%zu\n", size, dataset.c_str(), order, host_ms, elapsed_ms(setup), (haystack_len + 3 * needle_len) * 8);
        for (std::string algorithm : {"lower_bound", "upper_bound", "equal_range"}) {
          auto values = timings(stream, [&] {
            if (algorithm == "lower_bound" || algorithm == "equal_range") {
              thrust::lower_bound(policy, haystack, haystack + haystack_len, needles, needles + needle_len, begins, compare);
            }
            if (algorithm == "upper_bound") {
              thrust::upper_bound(policy, haystack, haystack + haystack_len, needles, needles + needle_len, begins, compare);
            } else if (algorithm == "equal_range") {
              // Thrust has no batched equal_range: explicitly compose its two batched bounds.
              thrust::upper_bound(policy, haystack, haystack + haystack_len, needles, needles + needle_len, ends, compare);
            }
          });
          check(begins, algorithm == "upper_bound" ? expected_ends : expected_begins);
          if (algorithm == "equal_range") check(ends, expected_ends);
          std::printf("implementation=cccl algorithm=%s size=%s dataset=%s order=%s key=u64 index=u64 haystack=%zu needles=%zu seed=%zu warmups=%d samples=%d stream=default resets=none best_ms=%.9f median_ms=%.9f sample_ms=", algorithm.c_str(), size, dataset.c_str(), order, haystack_len, needle_len, seed, warmups, samples, values.front(), values[samples / 2]);
          for (std::size_t i = 0; i < values.size(); ++i) std::printf("%s%.9f", i ? "," : "", values[i]);
          std::puts("");
        }
        CUDA_CHECK(cudaFree(ends));
        CUDA_CHECK(cudaFree(begins));
        CUDA_CHECK(cudaFree(needles));
        CUDA_CHECK(cudaFree(haystack));
      }
    }
  }
  return 0;
} catch (std::exception const& error) {
  std::fprintf(stderr, "Thrust sorted search reference failed: %s\n", error.what());
  return 1;
}
