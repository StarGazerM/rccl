#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>

#define CUDA_CHECK(call) do { cudaError_t status = (call); if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status)); } while (false)

namespace {
constexpr int warmups = 100;
constexpr int samples = 101;
constexpr int chains = 100;

template <int Tag>
__global__ void tiny(unsigned* output) {
  (void) Tag;
  if (threadIdx.x == 0) *output += 1;
}

void chain(unsigned* output) {
  tiny<0><<<1, 1>>>(output);
  tiny<1><<<1, 1>>>(output);
  tiny<2><<<1, 1>>>(output);
  tiny<3><<<1, 1>>>(output);
}
} // namespace

int main() try {
  unsigned* output{};
  CUDA_CHECK(cudaMalloc(&output, sizeof(unsigned)));
  CUDA_CHECK(cudaMemset(output, 0, sizeof(unsigned)));
  for (int i = 0; i < warmups; ++i) chain(output);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start{}, end{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));
  std::vector<float> timings;
  timings.reserve(samples);
  for (int sample = 0; sample < samples; ++sample) {
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < chains; ++i) chain(output);
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    float elapsed_ms{};
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, end));
    timings.push_back(elapsed_ms * 1000.0f / chains);
  }
  std::sort(timings.begin(), timings.end());

  unsigned actual{};
  CUDA_CHECK(cudaMemcpy(&actual, output, sizeof(unsigned), cudaMemcpyDeviceToHost));
  if (actual != static_cast<unsigned>((warmups + samples * chains) * 4)) {
    throw std::runtime_error("tiny-kernel chain result mismatch");
  }
  std::printf("implementation=cuda-runtime kernels=4 chains=%d median_us=%.6f\n",
              chains, timings[timings.size() / 2]);
  CUDA_CHECK(cudaEventDestroy(end));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaFree(output));
  return 0;
} catch (const std::exception& error) {
  std::fprintf(stderr, "CUDA submission probe failed: %s\n", error.what());
  return 1;
}
