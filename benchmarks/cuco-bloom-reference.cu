// Pinned oracle: NVIDIA/cuCollections 1d538b6befc286ab450d9e27d7ffbc4f1415c641.
#include <cuco/bloom_filter.cuh>
#include <cuco/bloom_filter_policy.cuh>
#include <cuco/extent.cuh>
#include <cuco/hash_functions.cuh>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_TRY(call)                                                                            \
  do {                                                                                            \
    cudaError_t error = (call);                                                                   \
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));                 \
  } while (false)

std::size_t argument(int argc, char** argv, char const* name, std::size_t fallback)
{
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string{argv[i]} == name) return std::stoull(argv[i + 1]);
  }
  return fallback;
}

std::uint64_t key(std::size_t index)
{
  std::uint64_t value = index + 1;
  value ^= value >> 30;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27;
  value *= 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

double median(std::vector<float> values)
{
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

int main(int argc, char** argv)
{
  try {
    auto const items      = argument(argc, argv, "--items", 8'000'000);
    auto const blocks     = argument(argc, argv, "--blocks", 10'000'000);
    auto const iterations = argument(argc, argv, "--iterations", 10);
    if (items == 0 || blocks == 0 || iterations == 0) throw std::runtime_error("invalid arguments");

    std::vector<std::uint64_t> host_keys(items);
    for (std::size_t i = 0; i < items; ++i) host_keys[i] = key(i);
    std::uint64_t* keys{};
    std::uint8_t* output{};
    CUDA_TRY(cudaMalloc(&keys, items * sizeof(*keys)));
    CUDA_TRY(cudaMalloc(&output, items * sizeof(*output)));
    CUDA_TRY(cudaMemcpy(keys, host_keys.data(), items * sizeof(*keys), cudaMemcpyHostToDevice));

    using hash_type   = cuco::xxhash_64<std::uint64_t>;
    using policy_type = cuco::bloom_filter_policy<std::uint64_t, hash_type>;
    using filter_type = cuco::bloom_filter<std::uint64_t,
                                           cuco::extent<std::size_t>,
                                           cuda::thread_scope_device,
                                           policy_type>;
    filter_type filter{cuco::extent<std::size_t>{blocks}, {}, policy_type{hash_type{0}}};

    for (int i = 0; i < 3; ++i) {
      filter.clear();
      filter.add(keys, keys + items);
      filter.contains(keys, keys + items, output);
    }

    cudaEvent_t start{}, end{};
    CUDA_TRY(cudaEventCreate(&start));
    CUDA_TRY(cudaEventCreate(&end));
    std::vector<float> add_times;
    add_times.reserve(iterations);
    for (std::size_t i = 0; i < iterations; ++i) {
      filter.clear();
      CUDA_TRY(cudaEventRecord(start));
      filter.add_async(keys, keys + items);
      CUDA_TRY(cudaEventRecord(end));
      CUDA_TRY(cudaEventSynchronize(end));
      float elapsed{};
      CUDA_TRY(cudaEventElapsedTime(&elapsed, start, end));
      add_times.push_back(elapsed);
    }

    std::vector<float> contains_times;
    contains_times.reserve(iterations);
    for (std::size_t i = 0; i < iterations; ++i) {
      CUDA_TRY(cudaEventRecord(start));
      filter.contains_async(keys, keys + items, output);
      CUDA_TRY(cudaEventRecord(end));
      CUDA_TRY(cudaEventSynchronize(end));
      float elapsed{};
      CUDA_TRY(cudaEventElapsedTime(&elapsed, start, end));
      contains_times.push_back(elapsed);
    }

    std::vector<std::uint8_t> host_output(items);
    CUDA_TRY(cudaMemcpy(host_output.data(), output, items * sizeof(*output), cudaMemcpyDeviceToHost));
    if (!std::all_of(host_output.begin(), host_output.end(), [](auto value) { return value == 1; })) {
      throw std::runtime_error("no-false-negative oracle failed");
    }
    std::vector<std::uint64_t> mixed_keys(items);
    for (std::size_t i = 0; i < items; ++i) {
      mixed_keys[i] = i % 2 == 0 ? host_keys[i] : key(i + items * 3 + 17);
    }
    CUDA_TRY(cudaMemcpy(keys, mixed_keys.data(), items * sizeof(*keys), cudaMemcpyHostToDevice));
    filter.contains(keys, keys + items, output);
    CUDA_TRY(cudaMemcpy(host_output.data(), output, items * sizeof(*output), cudaMemcpyDeviceToHost));
    std::uint64_t checksum = 0xcbf29ce484222325ULL;
    for (auto value : host_output) {
      checksum = (checksum ^ value) * 0x100000001b3ULL;
    }

    std::printf("implementation=cuCollections algorithm=bloom type=u64 items=%zu blocks=%zu "
                "checksum=%016llx add_median_ms=%.9f contains_median_ms=%.9f\n",
                items,
                blocks,
                static_cast<unsigned long long>(checksum),
                median(add_times),
                median(contains_times));
    CUDA_TRY(cudaEventDestroy(start));
    CUDA_TRY(cudaEventDestroy(end));
    CUDA_TRY(cudaFree(output));
    CUDA_TRY(cudaFree(keys));
    return 0;
  } catch (std::exception const& error) {
    std::fprintf(stderr, "cuCollections Bloom reference failed: %s\n", error.what());
    return 1;
  }
}
