#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>

#include "utils/check.hh"
#include "utils/util.hh"
#include "utils/fastexp.h"
#include "utils/fastlog.h"
#include "utils/fastgamma.h"

#ifndef COMPONENT_GAUSSIAN_HH_
#define COMPONENT_GAUSSIAN_HH_

template <typename T>
struct multi_gaussian_component_t {

  using Scalar   = typename T::Scalar;
  using Index    = typename T::Index;
  using mat_type = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
  using vec_type = Eigen::Matrix<Scalar, Eigen::Dynamic, 1, Eigen::ColMajor>;

  struct dim_t : public check_positive_t<Scalar> {
    explicit dim_t(const Index v) : check_positive_t<Scalar>(v) {}
  };

  explicit multi_gaussian_component_t(const dim_t _dim)
      : p(_dim.val),                // dimensionality
        d(static_cast<Scalar>(p)),  // dimensionality
        n(0.),                      // sample size
        s1(p, 0.),                  // 1st moment sum
        s2(0.),                     // 2nd moment sum
        scale(1.),                  // hyper for mean prior
        a0(1.),                     // hyper for precision prior
        b0(1.),                     // hyper for precision prior
        mu(p, 0.),                  // mean
        r(0.),                      // precision
        musq(0.) {                  // mean square
    // constructor
  }

  // log N(x|variational mean[x], variational var[mu] + variational var[x])
  template<typename Derived>
  Scalar lcvi(Eigen::MatrixBase<Derived>& xx) { // dim x 1
    const Derived& x = xx.derived();

    // (x - mu).cwiseProduct(x -mu)

    x.cwiseProduct(x);

  }


  // locally collapsed variational inference
  //   double score(const expr_vector_t& obs)
  //   {
  //     using namespace boost::math;

  //     const vec_type& x = obs.vec;

  //     double C = dot(x, x) + scale * dot(mu, mu);
  //     for (size_t j = 0; j < p; ++j)
  //     {
  //       double m = x.at(j) + scale * mu.at(j);
  //       C -= m * m / (1. + scale);
  //     }

  //     double ret = 0.5 * d * (std::log(scale) - std::log(1. + scale));
  //     ret += lgamma(a0 + 0.5 * d) - lgamma(a0);
  //     ret += a0 * std::log(b0) - (a0 + d * 0.5) * std::log(b0 + 0.5 * C);

  //     return ret;
  //   }

 private:
  const size_t p;  // dimensionality
  const Scalar d;  // dimensionality
  vec_type s1;     // sum_i z_i x_i
  Scalar s2;       // sum_i z_i <x_i,x_i>
  Scalar scale;    // hyper-parameter
  Scalar n;        // sum_i z_i
  Scalar a0;       // hyper for precision
  Scalar b0;       // hyper for precision
  vec_type mu;     // variational mu
  Scalar r;        // variational precision
  Scalar musq;     // E[mu^T mu]
};

#endif
