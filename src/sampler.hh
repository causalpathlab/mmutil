#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <random>

#ifndef SAMPLER_HH_
#define SAMPLER_HH_

template <typename Scalar, typename Index>
struct discrete_sampler_t {
  explicit discrete_sampler_t(const Index k)
      : K(k), maxval(0.0), exp_sumval(0.0), norm_exp(maxval) {}

  template <typename Derived>
  Index operator()(const Eigen::MatrixBase<Derived>& x_vec) {
    Index argmax_k;
    maxval     = x_vec.maxCoeff(&argmax_k);
    exp_sumval = x_vec.unaryExpr(norm_exp).sum();

    const Scalar u = Unif(Rng) * exp_sumval;
    Scalar cum     = 0.0;
    Index rIndex  = 0;
    for (auto k = 0; k < K; ++k) {
      const Scalar val = x_vec(k);
      cum += std::exp(val - maxval);
      if (u <= cum) {
        rIndex = k;
        break;
      }
    }
    return rIndex;
  }

  const Index K;

 private:
  Scalar maxval;
  Scalar exp_sumval;

  std::mt19937 Rng{std::random_device{}()};
  std::uniform_real_distribution<Scalar> Unif{0.0, 1.0};

  struct norm_exp_t {
    explicit norm_exp_t(const Scalar& _maxval) : maxval(_maxval) {}
    Scalar operator()(const Scalar& x) const { return std::exp(x - maxval); }
    const Scalar& maxval;
  } norm_exp;
};

#endif
