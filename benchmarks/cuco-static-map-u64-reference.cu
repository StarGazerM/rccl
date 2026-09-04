// Pinned cuCollections 1d538b6befc286ab450d9e27d7ffbc4f1415c641
// correctness oracle for vflog-style static_map<uint32_t, uint64_t> compressed ranges.

#include <cuco/static_map.cuh>

#include <cuda/iterator>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

std::uint64_t compress_u32(std::uint32_t high, std::uint32_t low)
{
  return (static_cast<std::uint64_t>(high) << 32) | static_cast<std::uint64_t>(low);
}

void check(cudaError_t status)
{
  if (status != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(status)); }
}

void expect(bool value, char const* message)
{
  if (!value) { throw std::runtime_error(message); }
}

int main()
{
  constexpr auto empty_key   = std::numeric_limits<std::uint32_t>::max();
  constexpr auto erased_key  = std::numeric_limits<std::uint32_t>::max() - 1;
  constexpr auto empty_value = static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max());

  std::vector<std::uint32_t> host_keys{7, 11, 19, 23, 29, 31, 37, 41};
  std::vector<std::uint64_t> host_values{
    compress_u32(0, 1),
    compress_u32(1, 2),
    compress_u32(0x7ffffffeU, 0xfffffffeU),
    compress_u32(0x80000000U, 3),
    compress_u32(0xfffffffeU, 0xfffffffeU),
    std::numeric_limits<std::uint64_t>::max() - 5,
    compress_u32(12345, 67890),
    (std::uint64_t{1} << 63) | 17};

  thrust::device_vector<std::uint32_t> keys(host_keys);
  thrust::device_vector<std::uint64_t> values(host_values);
  auto pairs = cuda::make_transform_iterator(
    cuda::counting_iterator<std::size_t>{0},
    cuda::proclaim_return_type<cuco::pair<std::uint32_t, std::uint64_t>>(
      [key_it = keys.begin(), value_it = values.begin()] __device__(std::size_t i) {
        return cuco::pair<std::uint32_t, std::uint64_t>{key_it[i], value_it[i]};
      }));

  using map_type = cuco::static_map<std::uint32_t, std::uint64_t>;
  auto map       = map_type{40, cuco::empty_key<std::uint32_t>{empty_key},
                            cuco::empty_value<std::uint64_t>{empty_value},
                            cuco::erased_key<std::uint32_t>{erased_key}};
  map.insert(pairs, pairs + host_keys.size());
  expect(map.size() == host_keys.size(), "u32/u64 oracle lost inserted keys");
  std::vector<std::uint32_t> query_host{31, 8, 7, 23, 41};
  thrust::device_vector<std::uint32_t> queries(query_host);
  thrust::device_vector<std::uint64_t> found(query_host.size());
  map.find(queries.begin(), queries.end(), found.begin());
  thrust::host_vector<std::uint64_t> host_found(found);
  std::vector<std::uint64_t> expected_found{host_values[5], empty_value, host_values[0],
                                            host_values[3], host_values[7]};
  expect(std::equal(host_found.begin(), host_found.end(), expected_found.begin()),
         "u32/u64 oracle find mismatch");

  thrust::device_vector<std::uint32_t> dup_key(1, 7);
  thrust::device_vector<std::uint64_t> dup_value(1, compress_u32(0xfffffffdU, 0xfffffffcU));
  auto assign_pair = cuda::make_transform_iterator(
    cuda::counting_iterator<std::size_t>{0},
    cuda::proclaim_return_type<cuco::pair<std::uint32_t, std::uint64_t>>(
      [key_it = dup_key.begin(), value_it = dup_value.begin()] __device__(std::size_t i) {
        return cuco::pair<std::uint32_t, std::uint64_t>{key_it[i], value_it[i]};
      }));
  map.insert_or_assign(assign_pair, assign_pair + 1);
  map.find(dup_key.begin(), dup_key.end(), found.begin());
  thrust::host_vector<std::uint64_t> assigned(found.begin(), found.begin() + 1);
  expect(assigned[0] == compress_u32(0xfffffffdU, 0xfffffffcU),
         "u32/u64 oracle assignment mismatch");

  thrust::device_vector<std::uint32_t> erase_keys(std::vector<std::uint32_t>{11, 23});
  map.erase(erase_keys.begin(), erase_keys.end());
  expect(map.size() == host_keys.size() - 2, "u32/u64 oracle erase mismatch");

  thrust::device_vector<std::uint32_t> out_keys(map.capacity());
  thrust::device_vector<std::uint64_t> out_values(map.capacity());
  auto [key_end, value_end] = map.retrieve_all(out_keys.begin(), out_values.begin());
  auto retrieved           = static_cast<std::size_t>(key_end - out_keys.begin());
  expect(retrieved == host_keys.size() - 2, "u32/u64 oracle retrieve_all count mismatch");

  std::cout << "Pinned cuCollections static_map<u32,u64> compressed-range oracle: PASS\n";
}
