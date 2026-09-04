// Pinned cuCollections 1d538b6befc286ab450d9e27d7ffbc4f1415c641
// equal-work reference for rccl-cuco's static_map<uint32_t, uint64_t> insertion path.

#include <cuco/static_map.cuh>

#include <cuda/iterator>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

std::uint32_t key(std::uint32_t index)
{
  auto value = index + 1;
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  value *= 0x846ca68bU;
  return value ^ (value >> 16);
}

std::uint64_t compressed_value(std::uint32_t index, std::uint32_t key)
{
  return ((0x80000000ULL | static_cast<std::uint64_t>(index)) << 32) |
         static_cast<std::uint64_t>(key * 17U);
}

void check(cudaError_t status)
{
  if (status != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(status)); }
}

int main(int argc, char** argv)
{
  std::size_t const items    = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 8'000'000;
  double const occupancy    = argc > 2 ? std::strtod(argv[2], nullptr) : 0.5;
  std::size_t const samples = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 10;
  if (items == 0 || items >= std::numeric_limits<std::uint32_t>::max() || occupancy <= 0.0 ||
      occupancy > 1.0 || samples == 0) {
    throw std::invalid_argument("invalid items, occupancy, or samples");
  }

  std::vector<std::uint32_t> host_keys(items);
  std::vector<std::uint64_t> host_values(items);
  for (std::size_t i = 0; i < items; ++i) {
    host_keys[i]   = key(static_cast<std::uint32_t>(i));
    host_values[i] = compressed_value(static_cast<std::uint32_t>(i), host_keys[i]);
  }
  thrust::device_vector<std::uint32_t> keys(host_keys);
  thrust::device_vector<std::uint64_t> values(host_values);
  auto pairs = cuda::make_transform_iterator(
    cuda::counting_iterator<std::size_t>{0},
    cuda::proclaim_return_type<cuco::pair<std::uint32_t, std::uint64_t>>(
      [key_it = keys.begin(), value_it = values.begin()] __device__(std::size_t i) {
        return cuco::pair<std::uint32_t, std::uint64_t>{key_it[i], value_it[i]};
      }));

  auto map = cuco::static_map<std::uint32_t, std::uint64_t>{
    static_cast<std::size_t>(items / occupancy),
    cuco::empty_key<std::uint32_t>{std::numeric_limits<std::uint32_t>::max()},
    cuco::empty_value<std::uint64_t>{static_cast<std::uint64_t>(
      std::numeric_limits<std::uint32_t>::max())},
    cuco::erased_key<std::uint32_t>{std::numeric_limits<std::uint32_t>::max() - 1}};

  for (int i = 0; i < 3; ++i) {
    map.clear();
    map.insert_async(pairs, pairs + items);
    check(cudaDeviceSynchronize());
  }

  cudaEvent_t start{};
  cudaEvent_t end{};
  check(cudaEventCreate(&start));
  check(cudaEventCreate(&end));
  std::vector<double> timings;
  timings.reserve(samples);
  for (std::size_t i = 0; i < samples; ++i) {
    map.clear();
    check(cudaEventRecord(start));
    map.insert_async(pairs, pairs + items);
    check(cudaEventRecord(end));
    check(cudaEventSynchronize(end));
    float milliseconds{};
    check(cudaEventElapsedTime(&milliseconds, start, end));
    timings.push_back(milliseconds);
  }
  check(cudaEventDestroy(start));
  check(cudaEventDestroy(end));

  if (map.size() != items) { throw std::runtime_error("reference insertion lost keys"); }
  std::sort(timings.begin(), timings.end());
  std::cout << std::fixed << std::setprecision(9)
            << "implementation=cuco algorithm=static_map_insert type=u32_u64 items=" << items
            << " occupancy=" << occupancy << " capacity=" << map.capacity()
            << " median_ms=" << timings[timings.size() / 2] << '\n';
}
