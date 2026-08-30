#include <thrust/execution_policy.h>
#include <thrust/set_operations.h>
#include <thrust/system/cuda/memory.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                                       \
  do                                                                                           \
  {                                                                                            \
    const cudaError_t status = (call);                                                         \
    if (status != cudaSuccess)                                                                 \
      throw std::runtime_error(cudaGetErrorString(status));                                    \
  } while (false)

struct timing
{
  double gpu_ms;
  double end_to_end_ms;
  std::size_t output_len;
};

// Exact-size single-stream cache matching pinned CCCL's caching_allocator_t.
struct cached_allocator
{
  using value_type = char;

  ~cached_allocator()
  {
    for (const auto& [bytes, pointer] : free_blocks)
    {
      (void) bytes;
      thrust::cuda::free(thrust::cuda::pointer<char>(pointer));
    }
    for (const auto& [pointer, bytes] : allocated_blocks)
    {
      (void) bytes;
      thrust::cuda::free(thrust::cuda::pointer<char>(pointer));
    }
  }

  char* allocate(std::ptrdiff_t bytes)
  {
    char* pointer{};
    const auto free = free_blocks.find(bytes);
    if (free == free_blocks.end())
      pointer = thrust::cuda::malloc<char>(bytes).get();
    else
    {
      pointer = free->second;
      free_blocks.erase(free);
    }
    allocated_blocks.emplace(pointer, bytes);
    return pointer;
  }

  void deallocate(char* pointer, std::size_t)
  {
    const auto allocated = allocated_blocks.find(pointer);
    if (allocated == allocated_blocks.end())
      throw std::runtime_error("temporary pointer was not allocated by cache");
    free_blocks.emplace(allocated->second, pointer);
    allocated_blocks.erase(allocated);
  }

private:
  std::multimap<std::ptrdiff_t, char*> free_blocks;
  std::map<char*, std::ptrdiff_t> allocated_blocks;
};

double median(std::vector<double> values)
{
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

template <class Operation>
timing timed(cudaStream_t stream, int iterations, Operation operation)
{
  std::size_t output_len = 0;
  for (int i = 0; i < 3; ++i)
    output_len = operation();

  cudaEvent_t start{}, end{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));
  std::vector<double> gpu;
  std::vector<double> wall;
  for (int i = 0; i < iterations; ++i)
  {
    const auto wall_start = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaEventRecord(start, stream));
    output_len = operation();
    CUDA_CHECK(cudaEventRecord(end, stream));
    CUDA_CHECK(cudaEventSynchronize(end));
    const auto wall_end = std::chrono::steady_clock::now();
    float elapsed = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, end));
    gpu.push_back(elapsed);
    wall.push_back(std::chrono::duration<double, std::milli>(wall_end - wall_start).count());
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(end));
  return {median(gpu), median(wall), output_len};
}

std::uint64_t splitmix64(std::uint64_t value)
{
  value += UINT64_C(0x9e3779b97f4a7c15);
  value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31);
}

template <class T>
std::vector<T> generate(std::size_t count, const std::string& entropy)
{
  const int rounds = entropy == "1.000" ? 1 : entropy == "0.201" ? 5 : 0;
  if (rounds == 0)
    throw std::runtime_error("unsupported Entropy axis");
  std::vector<T> result(count);
  for (std::size_t index = 0; index < count; ++index)
  {
    std::uint64_t bits = UINT64_MAX;
    for (int round = 0; round < rounds; ++round)
      bits &= splitmix64(UINT64_C(42) + index + (static_cast<std::uint64_t>(round) << 48));
    result[index] = static_cast<T>(bits);
  }
  return result;
}

void print_timings(
  const char* mode,
  const char* key,
  const char* value,
  std::size_t count,
  const std::string& entropy,
  std::size_t ratio,
  const timing& union_result,
  const timing& intersection_result,
  const timing& difference_result,
  const timing& symmetric_result)
{
  std::printf(
    "CCCL_THRUST_SET mode=%s key=%s value=%s items=%zu entropy=%s ratio=%zu "
    "union_gpu_median_ms=%.6f union_end_to_end_median_ms=%.6f "
    "intersection_gpu_median_ms=%.6f intersection_end_to_end_median_ms=%.6f "
    "difference_gpu_median_ms=%.6f difference_end_to_end_median_ms=%.6f "
    "symmetric_gpu_median_ms=%.6f symmetric_end_to_end_median_ms=%.6f\n",
    mode,
    key,
    value,
    count,
    entropy.c_str(),
    ratio,
    union_result.gpu_ms,
    union_result.end_to_end_ms,
    intersection_result.gpu_ms,
    intersection_result.end_to_end_ms,
    difference_result.gpu_ms,
    difference_result.end_to_end_ms,
    symmetric_result.gpu_ms,
    symmetric_result.end_to_end_ms);
}

template <class Key>
void run_keys(
  const char* key_name,
  std::size_t count,
  int iterations,
  const std::string& entropy,
  std::size_t ratio)
{
  const std::size_t first_count = ratio * count / 100;
  const std::size_t second_count = count - first_count;
  auto host = generate<Key>(count, entropy);
  std::sort(host.begin(), host.begin() + first_count);
  std::sort(host.begin() + first_count, host.end());

  Key *first{}, *second{}, *output{};
  CUDA_CHECK(cudaMalloc(&first, first_count * sizeof(Key)));
  CUDA_CHECK(cudaMalloc(&second, second_count * sizeof(Key)));
  CUDA_CHECK(cudaMalloc(&output, count * sizeof(Key)));
  CUDA_CHECK(cudaMemcpy(first, host.data(), first_count * sizeof(Key), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(second, host.data() + first_count, second_count * sizeof(Key), cudaMemcpyHostToDevice));

  cudaStream_t stream{};
  CUDA_CHECK(cudaStreamCreate(&stream));
  cached_allocator allocator;
  const auto policy = thrust::cuda::par(allocator).on(stream);
  const auto union_result = timed(stream, iterations, [&] {
    return static_cast<std::size_t>(
      thrust::set_union(policy, first, first + first_count, second, second + second_count, output) - output);
  });
  const auto intersection_result = timed(stream, iterations, [&] {
    return static_cast<std::size_t>(
      thrust::set_intersection(policy, first, first + first_count, second, second + second_count, output) - output);
  });
  const auto difference_result = timed(stream, iterations, [&] {
    return static_cast<std::size_t>(
      thrust::set_difference(policy, first, first + first_count, second, second + second_count, output) - output);
  });
  const auto symmetric_result = timed(stream, iterations, [&] {
    return static_cast<std::size_t>(
      thrust::set_symmetric_difference(policy, first, first + first_count, second, second + second_count, output)
      - output);
  });
  print_timings(
    "keys", key_name, "none", count, entropy, ratio, union_result, intersection_result, difference_result, symmetric_result);

  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(first));
  CUDA_CHECK(cudaFree(second));
  CUDA_CHECK(cudaFree(output));
}

template <class Key, class Value>
void run_pairs(
  const char* key_name,
  const char* value_name,
  std::size_t count,
  int iterations,
  const std::string& entropy,
  std::size_t ratio)
{
  const std::size_t first_count = ratio * count / 100;
  const std::size_t second_count = count - first_count;
  auto host = generate<Key>(count, entropy);
  std::sort(host.begin(), host.begin() + first_count);
  std::sort(host.begin() + first_count, host.end());

  Key *first_keys{}, *second_keys{}, *output_keys{};
  Value *first_values{}, *second_values{}, *output_values{};
  CUDA_CHECK(cudaMalloc(&first_keys, first_count * sizeof(Key)));
  CUDA_CHECK(cudaMalloc(&second_keys, second_count * sizeof(Key)));
  CUDA_CHECK(cudaMalloc(&output_keys, count * sizeof(Key)));
  CUDA_CHECK(cudaMalloc(&first_values, first_count * sizeof(Value)));
  CUDA_CHECK(cudaMalloc(&second_values, second_count * sizeof(Value)));
  CUDA_CHECK(cudaMalloc(&output_values, count * sizeof(Value)));
  CUDA_CHECK(cudaMemcpy(first_keys, host.data(), first_count * sizeof(Key), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(second_keys, host.data() + first_count, second_count * sizeof(Key), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(first_values, 0, first_count * sizeof(Value)));
  CUDA_CHECK(cudaMemset(second_values, 0, second_count * sizeof(Value)));

  cudaStream_t stream{};
  CUDA_CHECK(cudaStreamCreate(&stream));
  cached_allocator allocator;
  const auto policy = thrust::cuda::par(allocator).on(stream);
  const auto union_result = timed(stream, iterations, [&] {
    const auto end = thrust::set_union_by_key(
      policy,
      first_keys,
      first_keys + first_count,
      second_keys,
      second_keys + second_count,
      first_values,
      second_values,
      output_keys,
      output_values);
    return static_cast<std::size_t>(end.first - output_keys);
  });
  const auto intersection_result = timed(stream, iterations, [&] {
    const auto end = thrust::set_intersection_by_key(
      policy,
      first_keys,
      first_keys + first_count,
      second_keys,
      second_keys + second_count,
      first_values,
      output_keys,
      output_values);
    return static_cast<std::size_t>(end.first - output_keys);
  });
  const auto difference_result = timed(stream, iterations, [&] {
    const auto end = thrust::set_difference_by_key(
      policy,
      first_keys,
      first_keys + first_count,
      second_keys,
      second_keys + second_count,
      first_values,
      second_values,
      output_keys,
      output_values);
    return static_cast<std::size_t>(end.first - output_keys);
  });
  const auto symmetric_result = timed(stream, iterations, [&] {
    const auto end = thrust::set_symmetric_difference_by_key(
      policy,
      first_keys,
      first_keys + first_count,
      second_keys,
      second_keys + second_count,
      first_values,
      second_values,
      output_keys,
      output_values);
    return static_cast<std::size_t>(end.first - output_keys);
  });
  print_timings(
    "pairs",
    key_name,
    value_name,
    count,
    entropy,
    ratio,
    union_result,
    intersection_result,
    difference_result,
    symmetric_result);

  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(first_keys));
  CUDA_CHECK(cudaFree(second_keys));
  CUDA_CHECK(cudaFree(output_keys));
  CUDA_CHECK(cudaFree(first_values));
  CUDA_CHECK(cudaFree(second_values));
  CUDA_CHECK(cudaFree(output_values));
}

template <class Key>
void dispatch_value(
  const std::string& key,
  const std::string& value,
  std::size_t count,
  int iterations,
  const std::string& entropy,
  std::size_t ratio)
{
  if (value == "i8")
    run_pairs<Key, std::int8_t>(key.c_str(), "i8", count, iterations, entropy, ratio);
  else if (value == "i64")
    run_pairs<Key, std::int64_t>(key.c_str(), "i64", count, iterations, entropy, ratio);
  else
    throw std::runtime_error("unsupported ValueT axis");
}

int main(int argc, char** argv)
try
{
  const std::size_t count = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : (1u << 24);
  const int iterations = argc > 2 ? std::atoi(argv[2]) : 7;
  const std::string mode = argc > 3 ? argv[3] : "keys";
  const std::string key = argc > 4 ? argv[4] : "i32";
  const std::string value = argc > 5 ? argv[5] : "i64";
  const std::string entropy = argc > 6 ? argv[6] : "1.000";
  const std::size_t ratio = argc > 7 ? std::strtoull(argv[7], nullptr, 10) : 50;
  if (count < 2 || iterations < 1 || (ratio != 25 && ratio != 50 && ratio != 75))
    throw std::runtime_error("items, iterations, or SizeRatio is outside the pinned benchmark axes");

  if (mode == "keys")
  {
    if (key == "i8")
      run_keys<std::int8_t>("i8", count, iterations, entropy, ratio);
    else if (key == "i16")
      run_keys<std::int16_t>("i16", count, iterations, entropy, ratio);
    else if (key == "i32")
      run_keys<std::int32_t>("i32", count, iterations, entropy, ratio);
    else if (key == "i64")
      run_keys<std::int64_t>("i64", count, iterations, entropy, ratio);
    else
      throw std::runtime_error("unsupported KeyT axis");
  }
  else if (mode == "pairs")
  {
    if (key == "i8")
      dispatch_value<std::int8_t>(key, value, count, iterations, entropy, ratio);
    else if (key == "i16")
      dispatch_value<std::int16_t>(key, value, count, iterations, entropy, ratio);
    else if (key == "i32")
      dispatch_value<std::int32_t>(key, value, count, iterations, entropy, ratio);
    else if (key == "i64")
      dispatch_value<std::int64_t>(key, value, count, iterations, entropy, ratio);
    else
      throw std::runtime_error("unsupported KeyT axis");
  }
  else
    throw std::runtime_error("mode must be keys or pairs");
  return 0;
}
catch (const std::exception& error)
{
  std::fprintf(stderr, "%s\n", error.what());
  return 1;
}
