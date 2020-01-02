#include <algorithm>
#include <functional>

#include "inference/component_gaussian.hh"
#include "inference/dpm.hh"
#include "inference/sampler.hh"
#include "mmutil.hh"

#ifndef MMUTIL_CLUSTER_HH_
#define MMUTIL_CLUSTER_HH_

struct clustering_options_t {
  explicit clustering_options_t() {

    K           = 3;
    Alpha       = 1.0;
    burnin_iter = 10;
    max_iter    = 100;
    Tol         = 1e-4;
  }

  Index K;            // Truncation level
  Scalar Alpha;       // Truncated DPM prior
  Index burnin_iter;  // burn-in iterations
  Index max_iter;     // maximum number of iterations
  Scalar Tol;         // tolerance to check convergence
};

struct num_clust_t : public check_positive_t<Index> {
  explicit num_clust_t(const Index n) : check_positive_t<Index>(n) {}
};

struct num_sample_t : public check_positive_t<Index> {
  explicit num_sample_t(const Index n) : check_positive_t<Index>(n) {}
};

auto random_membership(const num_clust_t num_clust,  //
                       const num_sample_t num_sample) {

  std::random_device rd{};
  std::mt19937 gen{rd()};
  const Index k = num_clust.val;
  const Index n = num_sample.val;

  std::uniform_int_distribution<Index> runifK{0, k - 1};
  std::vector<Index> idx(n);
  std::iota(idx.begin(), idx.end(), 0);

  std::vector<Index> ret(n);
  ret.reserve(n);

  std::transform(idx.begin(), idx.end(), std::back_inserter(ret),
                 [&](const Index i) { return runifK(gen); });

  return ret;
}

template <typename F0, typename F>
void estimate_mixture_of_columns(const Mat& X, const clustering_options_t& options) {

  const Index K = options.K;
  const Index D = X.rows();
  const Index N = X.cols();
  typename F::dim_t dim(D);

  typename F0::dpm_alpha_t dpm_alpha(options.Alpha);
  typename F0::num_clust_t num_clust(K);
  using DS = discrete_sampler_t<Scalar, Index>;

  DS sampler_k(K);  // sample discrete from log-mass
  F0 prior(dpm_alpha, num_clust);

  std::vector<Index> cindex(K);
  std::iota(cindex.begin(), cindex.end(), 0);
  std::vector<F> components;
  std::transform(cindex.begin(), cindex.end(), std::back_inserter(components),
                 [&dim](const auto&) { return F(dim); });

  TLOG("Initialized " << K << " components");

  Vec mass(K);

  ////////////////////////////////////////////////////////
  // Kmeans++ initialization (Arthur and Vassilvitskii) //
  ////////////////////////////////////////////////////////

  std::vector<Index> membership(X.cols());
  {
    Vec x(D);
    x.setZero();
    DS sampler_n(N);
    Vec dist(N);

    for (Index k = 0; k < K; ++k) {
      dist          = (X.colwise() - x).cwiseProduct(X.colwise() - x).colwise().sum().transpose();
      dist          = dist.unaryExpr([](const Scalar _x) { return fasterlog(_x + 1e-8); });
      const Index j = sampler_n(dist);
      TLOG("Assigning " << j << " -> " << k);
      x = X.col(j).eval();
      components[k] += x;
      membership[j] = k;
      prior.add_to(k);
    }
  }

  for (Index i = 0; i < N; ++i) {
    if (membership.at(i) >= 0) continue;
    mass.setZero();
    for (Index k = 0; k < K; ++k) {
      mass(k) += components.at(k).log_lcvi(X.col(i));
    }
    const Index l = sampler_k(mass);
    membership[i] = l;
    prior.add_to(l);
    components[l] += X.col(i);
  }

  TLOG("Finished kmeans++ seeding");

  /////////////////////////////////
  // burn-in to initialize again //
  /////////////////////////////////
  {
    // progress_bar_t prog(options.burnin_iter, 1);

    for (Index b = 0; b < options.burnin_iter; ++b) {

      const Scalar temperature = options.burnin_iter - b;

      Scalar score = 0;
      for (Index i = 0; i < N; ++i) {
        Index k_old = membership.at(i);
        components[k_old] -= X.col(i);
        prior.subtract_from(k_old);

        mass.setZero();
        mass += prior.log_lcvi();
        for (Index k = 0; k < K; ++k) {
          mass(k) += components.at(k).log_lcvi(X.col(i));
        }

        Index k_new   = sampler_k(mass / temperature);
        membership[i] = k_new;

        prior.add_to(k_new);
        components[k_new] += X.col(i);

        score += components.at(k_new).log_marginal();
      }
      score /= static_cast<Scalar>(X.cols());
      // prog.update();
      // prog(std::cerr);
    }
  }

  for (auto k : membership) {
    std::cout << " " << k;
  }
  std::cout << std::endl;

  for (Index k = 0; k < K; ++k) {
    std::cout << components.at(k).posterior_mean().transpose() << std::endl;
  }

  ////////////////////////
  // variational update //
  ////////////////////////

  Mat Z(K, N);
  for (Index i = 0; i < N; ++i) {
    const Index k = membership.at(i);
    Z(k, i)       = 1.0;
  }

  const Scalar rate_discount = 0.55;

  Index t = 0;

  Scalar rate = std::pow(static_cast<Scalar>(t + 1), -rate_discount);

  ////////////////
  // Local step //
  ////////////////

  for (Index i = 0; i < N; ++i) {
    mass = prior.elbo();

    for (Index k = 0; k < K; ++k) {
      mass(k) += components.at(k).elbo(X.col(i));
    }

    const Scalar denom = log_sum_exp(mass);
    Z.col(i) = (mass.array() - denom).unaryExpr([](const Scalar x) { return fasterexp(x); });
  }

  /////////////////
  // Global step //
  /////////////////

  prior.update(Z, rate);

  //
}

//////////////////////////////////////////////////////
// A data-simulation routine for debugging purposes //
//////////////////////////////////////////////////////

inline auto simulate_gaussian_mixture(const Index n   = 100,     // sample size
                                      const Index p   = 2,       // dimension
                                      const Index k   = 3,       // #components
                                      const Scalar sd = 0.01) {  // jitter

  std::random_device rd{};
  std::mt19937 gen{rd()};
  std::normal_distribution<Scalar> rnorm{0, 1};

  // sample centers
  Mat centroid(p, k);  // dimension x cluster
  centroid = centroid.unaryExpr([&](const Scalar x) { return rnorm(gen); });

  // Index vector
  std::vector<Index> idx(n);
  std::iota(idx.begin(), idx.end(), 0);

  // sample membership
  auto membership = random_membership(num_clust_t(k), num_sample_t(n));

  SpMat Z(k, n);
  std::vector<Eigen::Triplet<Scalar> > _temp;
  _temp.reserve(n);
  for (Index i = 0; i < n; ++i) {
    _temp.push_back(Eigen::Triplet<Scalar>(membership.at(i), i, 1.0));
  }
  Z.reserve(n);
  Z.setFromTriplets(_temp.begin(), _temp.end());

  // sample data with random jittering
  Mat X(p, n);
  X = (centroid * Z).unaryExpr([&](const Scalar x) { return x + sd * rnorm(gen); }).eval();
  return std::make_tuple(X, membership, centroid);
}

#endif
