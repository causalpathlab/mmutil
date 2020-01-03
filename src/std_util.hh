#include <algorithm>
#include <execution>
#include <functional>

#ifndef STD_UTIL_HH_
#define STD_UTIL_HH_

template <typename Vec>
auto
std_argsort(const Vec& data) {
  using Index = std::ptrdiff_t;
  std::vector<Index> index(data.size());
  std::iota(std::begin(index), std::end(index), 0);
  std::sort(std::execution::seq, std::begin(index), std::end(index),
            [&](Index lhs, Index rhs) { return data.at(lhs) > data.at(rhs); });
  return index;
}

template <typename Vec>
auto
std_argsort_par(const Vec& data) {
  using Index = std::ptrdiff_t;
  std::vector<Index> index(data.size());
  std::iota(std::begin(index), std::end(index), 0);
  std::sort(std::execution::par, std::begin(index), std::end(index),
            [&](Index lhs, Index rhs) { return data.at(lhs) > data.at(rhs); });
  return index;
}

#endif
