#include <tuple>
#include <type_traits>
#include <utility>

#ifndef YPP_TUPLE_UTIL_HH_
#define YPP_TUPLE_UTIL_HH_

//////////////////////////////////////////////////////////////////
// apply function, one by one each, to each element of tuple    //
// e.g.,						        //
// func_apply(						        //
//   [](auto &&x) {					        //
//     std::cout << x.name << std::endl;		        //
//   },							        //
// std::make_tuple(eta_mean1, eta_mean2, eta_mean3, eta_var1)); //
//////////////////////////////////////////////////////////////////

template <typename Func, typename... Ts>
void
func_apply(Func &&func, std::tuple<Ts...> &&tup);

/////////////////////////////////////////////////////////////////
// create an arbitrary size tuples with lambda		       //
// e.g.,						       //
//   create_tuple<10>([](std::size_t j) { return obj.at(j); }) //
/////////////////////////////////////////////////////////////////

template <std::size_t N, typename Func>
auto
create_tuple(Func func);

/////////////////////
// implementations //
/////////////////////

template <typename Func, std::size_t... Is>
auto
create_tuple_impl(Func func, std::index_sequence<Is...>) {
  return std::make_tuple(func(Is)...);
}

template <std::size_t N, typename Func>
auto
create_tuple(Func func) {
  return create_tuple_impl(func, std::make_index_sequence<N>{});
}

////////////////////////////////////////////////////////////////
// apply function, one by one each, to each element of tuple
// 1. recurse
template <typename Func, typename Tuple, unsigned N>
struct func_apply_impl_t {
  static void run(Func &&f, Tuple &&tup) {
    func_apply_impl_t<Func, Tuple, N - 1>::run(std::forward<Func>(f), std::forward<Tuple>(tup));
    std::forward<Func>(f)(std::get<N>(std::forward<Tuple>(tup)));
  }
};

// 2. basecase
template <typename Func, typename Tuple>
struct func_apply_impl_t<Func, Tuple, 0> {
  static void run(Func &&f, Tuple &&tup) {
    std::forward<Func>(f)(std::get<0>(std::forward<Tuple>(tup)));
  }
};

template <typename Func, typename... Ts>
void
func_apply(Func &&f, std::tuple<Ts...> &&tup) {
  using Tuple = std::tuple<Ts...>;
  func_apply_impl_t<Func, Tuple, sizeof...(Ts) - 1>::run(std::forward<Func>(f),
                                                         std::forward<Tuple>(tup));
}

#endif
