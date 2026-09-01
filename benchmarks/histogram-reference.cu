#include <cub/device/device_histogram.cuh>

#include <cuda/std/array>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <vector>

#define CUDA_CHECK(call)                                                                    \
  do                                                                                        \
  {                                                                                         \
    const cudaError_t status = (call);                                                      \
    if (status != cudaSuccess)                                                              \
      throw std::runtime_error(cudaGetErrorString(status));                                 \
  } while (false)

std::vector<std::int32_t> random_bins(std::size_t count, std::int32_t bins)
{
  std::vector<std::int32_t> values(count);
  for (std::size_t index = 0; index < count; ++index)
  {
    std::uint64_t value = static_cast<std::uint64_t>(index) ^ 0x534d4152545f534full;
    value ^= value << 13;
    value ^= value >> 7;
    value ^= value << 17;
    values[index] = static_cast<std::int32_t>(value % static_cast<std::uint64_t>(bins));
  }
  return values;
}

std::vector<std::int32_t>
expected_channel(const std::vector<std::int32_t>& samples, std::int32_t bins, int channels, int channel)
{
  std::vector<std::int32_t> expected(static_cast<std::size_t>(bins));
  for (std::size_t index = static_cast<std::size_t>(channel); index < samples.size(); index += channels)
    ++expected[static_cast<std::size_t>(samples[index])];
  return expected;
}

void require_histogram(std::int32_t* histogram,
                       const std::vector<std::int32_t>& expected,
                       const char* operation)
{
  std::vector<std::int32_t> actual(expected.size());
  CUDA_CHECK(cudaMemcpy(actual.data(), histogram, actual.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost));
  if (actual != expected)
    throw std::runtime_error(operation);
}

template <class Operation>
double measure(cudaStream_t stream, cudaEvent_t start, cudaEvent_t end, int iterations, Operation operation)
{
  for (int i = 0; i < 3; ++i)
    CUDA_CHECK(operation());
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<float> samples;
  samples.reserve(static_cast<std::size_t>(iterations));
  for (int i = 0; i < iterations; ++i)
  {
    CUDA_CHECK(cudaEventRecord(start, stream));
    CUDA_CHECK(operation());
    CUDA_CHECK(cudaEventRecord(end, stream));
    CUDA_CHECK(cudaEventSynchronize(end));
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, end));
    samples.push_back(milliseconds);
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

int main(int argc, char** argv)
try
{
  const std::int64_t items = argc > 1 ? std::strtoll(argv[1], nullptr, 10) : 16'777'216;
  const std::int32_t bins  = argc > 2 ? std::atoi(argv[2]) : 128;
  const int iterations     = argc > 3 ? std::atoi(argv[3]) : 15;
  if (items <= 0 || bins <= 0 || iterations <= 0)
    throw std::runtime_error("items, bins, and iterations must be positive");

  constexpr int channels        = 4;
  constexpr int active_channels = 3;
  const auto multi_items         = static_cast<std::size_t>(items) * channels;
  auto single_host               = random_bins(static_cast<std::size_t>(items), bins);
  auto multi_host                = random_bins(multi_items, bins);
  const auto single_expected     = expected_channel(single_host, bins, 1, 0);
  const auto multi_expected      = cuda::std::array<std::vector<std::int32_t>, active_channels>{
    expected_channel(multi_host, bins, channels, 0),
    expected_channel(multi_host, bins, channels, 1),
    expected_channel(multi_host, bins, channels, 2)};
  std::vector<std::int32_t> levels(static_cast<std::size_t>(bins) + 1);
  std::iota(levels.begin(), levels.end(), 0);

  cudaStream_t stream{};
  CUDA_CHECK(cudaStreamCreate(&stream));
  std::int32_t* single_samples = nullptr;
  std::int32_t* multi_samples  = nullptr;
  std::int32_t* d_levels       = nullptr;
  std::int32_t* single_hist    = nullptr;
  cuda::std::array<std::int32_t*, active_channels> multi_hist{};
  CUDA_CHECK(cudaMalloc(&single_samples, single_host.size() * sizeof(std::int32_t)));
  CUDA_CHECK(cudaMalloc(&multi_samples, multi_host.size() * sizeof(std::int32_t)));
  CUDA_CHECK(cudaMalloc(&d_levels, levels.size() * sizeof(std::int32_t)));
  CUDA_CHECK(cudaMalloc(&single_hist, static_cast<std::size_t>(bins) * sizeof(std::int32_t)));
  for (auto& channel : multi_hist)
    CUDA_CHECK(cudaMalloc(&channel, static_cast<std::size_t>(bins) * sizeof(std::int32_t)));
  CUDA_CHECK(cudaMemcpyAsync(single_samples,
                             single_host.data(),
                             single_host.size() * sizeof(std::int32_t),
                             cudaMemcpyHostToDevice,
                             stream));
  CUDA_CHECK(cudaMemcpyAsync(multi_samples,
                             multi_host.data(),
                             multi_host.size() * sizeof(std::int32_t),
                             cudaMemcpyHostToDevice,
                             stream));
  CUDA_CHECK(cudaMemcpyAsync(d_levels,
                             levels.data(),
                             levels.size() * sizeof(std::int32_t),
                             cudaMemcpyHostToDevice,
                             stream));

  const auto level_counts = cuda::std::array<int, active_channels>{bins + 1, bins + 1, bins + 1};
  const auto lower_levels = cuda::std::array<std::int32_t, active_channels>{0, 0, 0};
  const auto upper_levels = cuda::std::array<std::int32_t, active_channels>{bins, bins, bins};
  const auto level_ptrs =
    cuda::std::array<const std::int32_t*, active_channels>{d_levels, d_levels, d_levels};

  std::size_t even_bytes = 0;
  std::size_t range_bytes = 0;
  std::size_t multi_even_bytes = 0;
  std::size_t multi_range_bytes = 0;
  CUDA_CHECK(cub::DeviceHistogram::HistogramEven(nullptr,
                                                  even_bytes,
                                                  single_samples,
                                                  single_hist,
                                                  bins + 1,
                                                  0,
                                                  bins,
                                                  items,
                                                  stream));
  CUDA_CHECK(cub::DeviceHistogram::HistogramRange(nullptr,
                                                   range_bytes,
                                                   single_samples,
                                                   single_hist,
                                                   bins + 1,
                                                   d_levels,
                                                   items,
                                                   stream));
  CUDA_CHECK((cub::DeviceHistogram::MultiHistogramEven<channels, active_channels>(nullptr,
                                                                                   multi_even_bytes,
                                                                                   multi_samples,
                                                                                   multi_hist,
                                                                                   level_counts,
                                                                                   lower_levels,
                                                                                   upper_levels,
                                                                                   items,
                                                                                   stream)));
  CUDA_CHECK((cub::DeviceHistogram::MultiHistogramRange<channels, active_channels>(nullptr,
                                                                                    multi_range_bytes,
                                                                                    multi_samples,
                                                                                    multi_hist,
                                                                                    level_counts,
                                                                                    level_ptrs,
                                                                                    items,
                                                                                    stream)));
  const std::size_t temp_bytes =
    std::max({even_bytes, range_bytes, multi_even_bytes, multi_range_bytes});
  void* temporary = nullptr;
  CUDA_CHECK(cudaMalloc(&temporary, temp_bytes));

  cudaEvent_t start{}, end{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));
  const double even_ms = measure(stream, start, end, iterations, [&] {
    return cub::DeviceHistogram::HistogramEven(temporary,
                                                even_bytes,
                                                single_samples,
                                                single_hist,
                                                bins + 1,
                                                0,
                                                bins,
                                                items,
                                                stream);
  });
  require_histogram(single_hist, single_expected, "CCCL HistogramEven validation failed");
  const double range_ms = measure(stream, start, end, iterations, [&] {
    return cub::DeviceHistogram::HistogramRange(temporary,
                                                 range_bytes,
                                                 single_samples,
                                                 single_hist,
                                                 bins + 1,
                                                 d_levels,
                                                 items,
                                                 stream);
  });
  require_histogram(single_hist, single_expected, "CCCL HistogramRange validation failed");
  const double multi_even_ms = measure(stream, start, end, iterations, [&] {
    return cub::DeviceHistogram::MultiHistogramEven<channels, active_channels>(temporary,
                                                                                multi_even_bytes,
                                                                                multi_samples,
                                                                                multi_hist,
                                                                                level_counts,
                                                                                lower_levels,
                                                                                upper_levels,
                                                                                items,
                                                                                stream);
  });
  for (int channel = 0; channel < active_channels; ++channel)
    require_histogram(multi_hist[channel], multi_expected[channel], "CCCL MultiHistogramEven validation failed");
  const double multi_range_ms = measure(stream, start, end, iterations, [&] {
    return cub::DeviceHistogram::MultiHistogramRange<channels, active_channels>(temporary,
                                                                                 multi_range_bytes,
                                                                                 multi_samples,
                                                                                 multi_hist,
                                                                                 level_counts,
                                                                                 level_ptrs,
                                                                                 items,
                                                                                 stream);
  });
  for (int channel = 0; channel < active_channels; ++channel)
    require_histogram(multi_hist[channel], multi_expected[channel], "CCCL MultiHistogramRange validation failed");

  std::printf("CCCL_CUB_HISTOGRAM items=%lld bins=%d even_gpu_median_ms=%.9f "
              "range_gpu_median_ms=%.9f multi_even_gpu_median_ms=%.9f "
              "multi_range_gpu_median_ms=%.9f\n",
              static_cast<long long>(items),
              bins,
              even_ms,
              range_ms,
              multi_even_ms,
              multi_range_ms);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(end));
  CUDA_CHECK(cudaFree(temporary));
  CUDA_CHECK(cudaFree(single_samples));
  CUDA_CHECK(cudaFree(multi_samples));
  CUDA_CHECK(cudaFree(d_levels));
  CUDA_CHECK(cudaFree(single_hist));
  for (auto channel : multi_hist)
    CUDA_CHECK(cudaFree(channel));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
catch (const std::exception& error)
{
  std::fprintf(stderr, "%s\n", error.what());
  return 1;
}
