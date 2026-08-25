#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
constexpr std::size_t alignment{256};
constexpr std::string_view rmm_commit{"c65202f52766e12c73e293b16d6a520a069d4afc"};

struct operation {
  bool allocate;
  std::size_t slot;
  std::size_t size;
};

class lcg {
 public:
  explicit lcg(std::uint64_t seed) : state_{seed} {}

  std::uint64_t next()
  {
    state_ = state_ * UINT64_C(6364136223846793005) + UINT64_C(1);
    return state_;
  }

  std::size_t below(std::size_t maximum)
  {
    return static_cast<std::size_t>(next()) % maximum;
  }

 private:
  std::uint64_t state_;
};

std::vector<operation> make_trace(std::size_t allocation_count,
                                  std::size_t maximum_allocation,
                                  std::size_t maximum_live_bytes,
                                  std::uint64_t seed)
{
  lcg random{seed};
  std::vector<operation> operations;
  operations.reserve(allocation_count * 2);
  std::vector<std::pair<std::size_t, std::size_t>> active;
  std::size_t live_bytes{};
  std::size_t allocated{};
  while (allocated < allocation_count) {
    auto const size = random.below(maximum_allocation) + 1;
    auto const should_allocate =
      active.empty() ||
      (random.below(100) < 73 && live_bytes + size <= maximum_live_bytes);
    if (should_allocate) {
      auto const slot = allocated;
      operations.push_back({true, slot, size});
      active.emplace_back(slot, size);
      live_bytes += size;
      ++allocated;
    } else {
      auto const index        = random.below(active.size());
      auto const [slot, size] = active[index];
      active[index]           = active.back();
      active.pop_back();
      operations.push_back({false, slot, 0});
      live_bytes -= size;
    }
  }
  while (!active.empty()) {
    auto const index  = random.below(active.size());
    auto const slot   = active[index].first;
    active[index]     = active.back();
    active.pop_back();
    operations.push_back({false, slot, 0});
  }
  return operations;
}

struct trace_result {
  std::uint64_t elapsed_ns;
  std::vector<std::uint64_t> latencies;
};

template <typename Resource>
trace_result run_trace(Resource& resource,
                       rmm::cuda_stream_view stream,
                       std::vector<operation> const& operations,
                       std::size_t allocation_count,
                       bool capture_latency)
{
  std::vector<void*> live(allocation_count, nullptr);
  std::vector<std::size_t> live_sizes(allocation_count, 0);
  std::vector<std::uint64_t> latencies;
  if (capture_latency) { latencies.reserve(operations.size()); }
  auto const started = std::chrono::steady_clock::now();
  for (auto const& op : operations) {
    auto const operation_started = std::chrono::steady_clock::now();
    if (op.allocate) {
      live[op.slot]       = resource.allocate(stream, op.size, alignment);
      live_sizes[op.slot] = op.size;
    } else {
      resource.deallocate(stream, live[op.slot], live_sizes[op.slot], alignment);
      live[op.slot]       = nullptr;
      live_sizes[op.slot] = 0;
    }
    if (capture_latency) {
      auto const elapsed = std::chrono::steady_clock::now() - operation_started;
      latencies.push_back(
        static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed)
                                     .count()));
    }
  }
  stream.synchronize();
  auto const elapsed = std::chrono::steady_clock::now() - started;
  return {static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()),
          std::move(latencies)};
}

std::size_t argument(int argc, char** argv, std::string_view name, std::size_t default_value)
{
  auto const prefix = std::string{"--"} + std::string{name} + "=";
  for (int index = 1; index < argc; ++index) {
    std::string_view value{argv[index]};
    if (value.starts_with(prefix)) {
      return static_cast<std::size_t>(std::stoull(std::string{value.substr(prefix.size())}));
    }
  }
  return default_value;
}

std::string resource_name(int argc, char** argv)
{
  constexpr std::string_view prefix{"--resource="};
  for (int index = 1; index < argc; ++index) {
    std::string_view value{argv[index]};
    if (value.starts_with(prefix)) { return std::string{value.substr(prefix.size())}; }
  }
  return "pool";
}

std::uint64_t percentile(std::vector<std::uint64_t> const& sorted, double value)
{
  auto const index = static_cast<std::size_t>(
    std::llround(static_cast<double>(sorted.size() - 1) * value));
  return sorted[index];
}

template <typename Resource>
int benchmark(Resource& resource,
              std::string const& resource_name,
              std::vector<operation> const& operations,
              std::size_t allocation_count,
              std::size_t maximum_allocation,
              std::size_t maximum_live_bytes,
              std::size_t warmups,
              std::size_t repetitions,
              std::uint64_t seed)
{
  rmm::cuda_stream_view stream{};
  for (std::size_t index = 0; index < warmups; ++index) {
    (void)run_trace(resource, stream, operations, allocation_count, false);
  }

  std::vector<std::uint64_t> elapsed;
  std::vector<std::uint64_t> latency;
  elapsed.reserve(repetitions);
  for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
    auto result = run_trace(resource,
                            stream,
                            operations,
                            allocation_count,
                            repetition == repetitions - 1);
    elapsed.push_back(result.elapsed_ns);
    latency.insert(latency.end(), result.latencies.begin(), result.latencies.end());
  }
  std::ranges::sort(elapsed);
  std::ranges::sort(latency);
  auto const median_ns = percentile(elapsed, 0.5);
  auto const operations_per_second =
    static_cast<double>(operations.size()) / (static_cast<double>(median_ns) / 1.0e9);

  std::cout << "{\"schema\":1,\"rmm_commit\":\"" << rmm_commit << "\",\"resource\":\""
            << resource_name << "\",\"seed\":" << seed << ",\"allocations\":"
            << allocation_count << ",\"operations\":" << operations.size()
            << ",\"max_allocation_bytes\":" << maximum_allocation << ",\"max_live_bytes\":"
            << maximum_live_bytes << ",\"warmups\":" << warmups << ",\"repetitions\":"
            << repetitions << ",\"median_trace_ns\":" << median_ns
            << ",\"operations_per_second\":" << operations_per_second
            << ",\"p50_operation_ns\":" << percentile(latency, 0.5)
            << ",\"p99_operation_ns\":" << percentile(latency, 0.99) << "}\n";
  return EXIT_SUCCESS;
}
}  // namespace

int main(int argc, char** argv)
try {
  auto const allocation_count   = argument(argc, argv, "allocations", 10'000);
  auto const maximum_allocation = argument(argc, argv, "max-allocation", 1 << 20);
  auto const maximum_live_bytes = argument(argc, argv, "max-live-bytes", 256 << 20);
  auto const warmups             = argument(argc, argv, "warmups", 2);
  auto const repetitions         = argument(argc, argv, "repetitions", 7);
  auto const seed = static_cast<std::uint64_t>(argument(argc, argv, "seed", 0x5eed));
  auto const name = resource_name(argc, argv);
  if (allocation_count == 0 || maximum_allocation == 0 || maximum_live_bytes == 0 ||
      repetitions == 0) {
    throw std::invalid_argument{"benchmark sizes and repetitions must be non-zero"};
  }
  auto const operations =
    make_trace(allocation_count, maximum_allocation, maximum_live_bytes, seed);

  if (name == "pool") {
    auto const initial_size = std::min(maximum_live_bytes, std::size_t{64} << 20);
    rmm::mr::pool_memory_resource resource{
      rmm::mr::cuda_memory_resource{}, initial_size, std::nullopt};
    return benchmark(resource,
                     name,
                     operations,
                     allocation_count,
                     maximum_allocation,
                     maximum_live_bytes,
                     warmups,
                     repetitions,
                     seed);
  }
  if (name == "cuda_async") {
    rmm::mr::cuda_async_memory_resource resource{};
    return benchmark(resource,
                     name,
                     operations,
                     allocation_count,
                     maximum_allocation,
                     maximum_live_bytes,
                     warmups,
                     repetitions,
                     seed);
  }
  if (name == "cuda") {
    rmm::mr::cuda_memory_resource resource{};
    return benchmark(resource,
                     name,
                     operations,
                     allocation_count,
                     maximum_allocation,
                     maximum_live_bytes,
                     warmups,
                     repetitions,
                     seed);
  }
  throw std::invalid_argument{"unknown resource " + name};
} catch (std::exception const& error) {
  std::cerr << "rmm_allocation_trace: " << error.what() << '\n';
  return EXIT_FAILURE;
}
