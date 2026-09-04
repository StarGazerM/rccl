// Pinned cuCollections 1d538b6befc286ab450d9e27d7ffbc4f1415c641
// equal-work reference for rccl-cuco's exact HISA Row<2> set.

#include <cuco/hash_functions.cuh>
#include <cuco/probing_scheme.cuh>
#include <cuco/static_set.cuh>

#include <cuda/std/functional>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

struct Row2 {
  std::uint32_t data[2];
  __host__ __device__ friend bool operator==(Row2 const& lhs, Row2 const& rhs)
  {
    return lhs.data[0] == rhs.data[0] && lhs.data[1] == rhs.data[1];
  }
};
static_assert(sizeof(Row2) == 8);

Row2 row(std::uint32_t index)
{
  auto value = index + 1;
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  value *= 0x846ca68bU;
  value ^= value >> 16;
  return Row2{{value, value ^ 0x9e3779b9U}};
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
  std::string const operation = argc > 4 ? argv[4] : "insert";
  if (items == 0 || items >= std::numeric_limits<std::uint32_t>::max() || occupancy <= 0.0 ||
      occupancy > 1.0 || samples == 0 || (operation != "insert" && operation != "contains")) {
    throw std::invalid_argument("invalid benchmark arguments");
  }

  std::vector<Row2> host_rows(items);
  for (std::size_t i = 0; i < items; ++i) host_rows[i] = row(static_cast<std::uint32_t>(i));
  thrust::device_vector<Row2> rows(host_rows);
  thrust::device_vector<std::uint32_t> output(items);
  auto* row_ptr = thrust::raw_pointer_cast(rows.data());
  auto* output_ptr = thrust::raw_pointer_cast(output.data());

  using hasher = cuco::xxhash_64<Row2>;
  using probe = cuco::double_hashing<4, hasher>;
  using set_type = cuco::static_set<Row2,
                                    cuco::extent<std::size_t>,
                                    cuda::thread_scope_device,
                                    cuda::std::equal_to<Row2>,
                                    probe>;
  auto const requested = static_cast<std::size_t>(items / occupancy);
  set_type set{requested,
               cuco::empty_key<Row2>{Row2{{std::numeric_limits<std::uint32_t>::max(),
                                          std::numeric_limits<std::uint32_t>::max()}}},
               {}, probe{hasher{0}, hasher{1}}};

  if (operation == "contains") {
    set.insert_async(row_ptr, row_ptr + items);
    check(cudaDeviceSynchronize());
  }
  for (int i = 0; i < 3; ++i) {
    if (operation == "insert") {
      set.clear();
      set.insert_async(row_ptr, row_ptr + items);
    } else {
      set.contains_async(row_ptr, row_ptr + items, output_ptr);
    }
    check(cudaDeviceSynchronize());
  }

  cudaEvent_t start{};
  cudaEvent_t end{};
  check(cudaEventCreate(&start));
  check(cudaEventCreate(&end));
  std::vector<double> timings;
  timings.reserve(samples);
  for (std::size_t i = 0; i < samples; ++i) {
    if (operation == "insert") set.clear();
    check(cudaEventRecord(start));
    if (operation == "insert") {
      set.insert_async(row_ptr, row_ptr + items);
    } else {
      set.contains_async(row_ptr, row_ptr + items, output_ptr);
    }
    check(cudaEventRecord(end));
    check(cudaEventSynchronize(end));
    float milliseconds{};
    check(cudaEventElapsedTime(&milliseconds, start, end));
    timings.push_back(milliseconds);
  }
  check(cudaEventDestroy(start));
  check(cudaEventDestroy(end));

  if (operation == "insert" && set.size() != items) {
    throw std::runtime_error("reference insertion lost rows");
  }
  if (operation == "contains") {
    std::vector<std::uint32_t> host_output(items);
    thrust::copy(output.begin(), output.end(), host_output.begin());
    if (!std::all_of(host_output.begin(), host_output.end(), [](auto value) { return value == 1; })) {
      throw std::runtime_error("reference contains missed rows");
    }
  }
  std::sort(timings.begin(), timings.end());
  std::cout << std::fixed << std::setprecision(9)
            << "implementation=cuco algorithm=exact_row_set_" << operation
            << " type=row2_u32 items=" << items << " occupancy=" << occupancy
            << " capacity=" << set.capacity()
            << " median_ms=" << timings[timings.size() / 2] << '\n';
}
