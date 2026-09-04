// Pinned CCCL Row<6> resource oracle for in-place and out-of-place stable key sort.
#include <cub/device/device_merge_sort.cuh>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

struct alignas(4) Row6 {
  std::uint32_t fields[6];
};

static_assert(sizeof(Row6) == 24);
static_assert(alignof(Row6) == 4);

struct CompareRow6 {
  __host__ __device__ bool operator()(Row6 const& left, Row6 const& right) const {
#pragma unroll
    for (int column = 0; column < 6; ++column) {
      if (left.fields[column] < right.fields[column]) return true;
      if (left.fields[column] > right.fields[column]) return false;
    }
    return false;
  }
};

int main() {
  constexpr int items = 513;
  Row6* input = nullptr;
  Row6* output = nullptr;
  void* temporary = nullptr;
  std::size_t temporary_bytes = 0;

  cudaMalloc(&input, sizeof(Row6) * items);
  cudaMalloc(&output, sizeof(Row6) * items);

  cub::DeviceMergeSort::StableSortKeys(
      nullptr, temporary_bytes, input, items, CompareRow6{});
  cudaMalloc(&temporary, temporary_bytes);
  cub::DeviceMergeSort::StableSortKeys(
      temporary, temporary_bytes, input, items, CompareRow6{});
  cudaFree(temporary);

  temporary = nullptr;
  temporary_bytes = 0;
  cub::DeviceMergeSort::StableSortKeysCopy(
      nullptr, temporary_bytes, input, output, items, CompareRow6{});
  cudaMalloc(&temporary, temporary_bytes);
  cub::DeviceMergeSort::StableSortKeysCopy(
      temporary, temporary_bytes, input, output, items, CompareRow6{});

  cudaDeviceSynchronize();
  cudaFree(temporary);
  cudaFree(output);
  cudaFree(input);
}
