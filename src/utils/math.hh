#include "utils/fastexp.h"
#include "utils/fastlog.h"

#ifndef _UTIL_MATH_HH_
#define _UTIL_MATH_HH_

/////////////////////
// log(1 + exp(x)) //
/////////////////////

template <typename T>
inline T _softplus(const T x) {
  const T cutoff = static_cast<T>(10.);
  const T one = static_cast<T>(1.0);
  if (x > cutoff) {
    return x + fasterlog(one + fasterexp(-x));
  }
  return fasterlog(one + fasterexp(x));
}

//////////////////////////
// log(exp(a) + exp(b)) //
//////////////////////////

template <typename T>
inline T _log_sum_exp(const T log_a, const T log_b) {
  const T one = static_cast<T>(1.0);
  if (log_a > log_b) {
    return log_a + _softplus(log_b - log_a);
  }
  return log_b + _softplus(log_a - log_b);
}

#endif
