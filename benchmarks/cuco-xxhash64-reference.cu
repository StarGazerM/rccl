// Pinned cuCollections 1d538b6befc286ab450d9e27d7ffbc4f1415c641
// XXHash_64 oracle for SRDatalog TVJoin/join_skew/join_include.cuh TBlob rows.

#include <cuco/hash_functions.cuh>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>

template <std::size_t N>
struct HisaRow {
  std::uint32_t data[N];
};

void print_hash(char const* kind,
                std::size_t length,
                std::uint64_t seed,
                std::uint64_t hash)
{
  std::cout << kind << ' ' << length << " 0x" << std::hex << std::setw(16)
            << std::setfill('0') << seed << " 0x" << std::setw(16) << hash << std::dec << '\n';
}

template <typename T>
void print_value(char const* kind, T const& value, std::uint64_t seed)
{
  print_hash(kind, sizeof(T), seed, cuco::xxhash_64<T>{seed}(value));
}

int main()
{
  constexpr std::array<std::uint64_t, 3> seeds{
    0,
    1,
    0x9e3779b97f4a7c15ULL,
  };
  alignas(8) std::array<std::uint8_t, 64> aligned{};
  std::array<std::uint8_t, 65> unaligned{};
  for (std::size_t index = 0; index < aligned.size(); ++index) {
    auto const value     = static_cast<std::uint8_t>(index * 37 + 11);
    aligned[index]       = value;
    unaligned[index + 1] = value;
  }

  for (std::size_t length = 0; length <= 63; ++length) {
    auto const seed = seeds[length % seeds.size()];
    auto const hash = cuco::xxhash_64<std::uint8_t>{seed};
    print_hash("tail-aligned",
               length,
               seed,
               hash.compute_hash(reinterpret_cast<std::byte const*>(aligned.data()), length));
    print_hash("tail-unaligned",
               length,
               seed,
               hash.compute_hash(reinterpret_cast<std::byte const*>(unaligned.data() + 1), length));
  }

  std::uint32_t const u32 = 0x12345678U;
  std::uint64_t const u64 = 0x0123456789abcdefULL;
  print_value("u32", u32, 0);
  print_value("u64", u64, 0x9e3779b97f4a7c15ULL);

  HisaRow<1> const row1{{0}};
  HisaRow<2> const row2{{1, 2}};
  HisaRow<4> const row4{{0, 1, 0x12345678U, 0xffffffffU}};
  HisaRow<7> const row7{{7, 0, 42, 0x80000000U, 3, 99, 0xffffffffU}};
  print_value("hisa-row-1", row1, 0);
  print_value("hisa-row-2", row2, 1);
  print_value("hisa-row-4", row4, 0x9e3779b97f4a7c15ULL);
  print_value("hisa-row-7", row7, 0);
}
