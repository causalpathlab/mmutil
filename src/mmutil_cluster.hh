#include <getopt.h>

#include <algorithm>
#include <functional>
#include <string>
#include <unordered_set>

#include "inference/component_gaussian.hh"
#include "inference/dpm.hh"
#include "inference/sampler.hh"
#include "mmutil.hh"
#include "mmutil_match.hh"
#include "mmutil_normalize.hh"
#include "mmutil_spectral.hh"
#include "utils/graph.hh"
#include "utils/progress.hh"

#ifndef MMUTIL_CLUSTER_HH_
#define MMUTIL_CLUSTER_HH_

struct cluster_options_t {

  typedef enum { DBSCAN, GAUSSIAN_MIXTURE } method_t;
  const std::vector<std::string> METHOD_NAMES;

  typedef enum { UNIFORM, CV, MEAN } sampling_method_t;
  const std::vector<std::string> SAMPLING_METHOD_NAMES;

  explicit cluster_options_t()
      : METHOD_NAMES{"DBSCAN", "GMM"},
        SAMPLING_METHOD_NAMES{"UNIFORM", "CV", "MEAN"} {

    K             = 3;
    Alpha         = 1.0;
    burnin_iter   = 10;
    max_iter      = 100;
    min_iter      = 5;
    Tol           = 1e-4;
    rate_discount = .55;
    bilink        = 10;
    nlist         = 10;
    kmeanspp      = false;
    knn           = 50;
    knn_cutoff    = 1.0;
    levels        = 10;

    out      = "output";
    out_data = false;

    method = DBSCAN;

    tau              = 1.0;
    rank             = 10;
    lu_iter          = 3;
    embedding_dim    = 2;
    embedding_epochs = 20;
    min_size         = 10;
    prune_knn        = false;
    raw_scale        = false;
    log_scale        = true;
    col_norm         = 10000;

    row_weight_file = "";

    nystrom_sample = 10000;
    nystrom_batch  = 10000;

    nystrom_sample_method = UNIFORM;

    verbose = false;
  }

  Index K;               // Truncation level
  Scalar Alpha;          // Truncated DPM prior
  Index burnin_iter;     // burn-in iterations
  Index max_iter;        // maximum number of iterations
  Index min_iter;        // minimum number of iterations
  Scalar Tol;            // tolerance to check convergence
  Scalar rate_discount;  // learning rate discount
  Index bilink;
  Index nlist;

  std::string out;
  std::string spectral_file;
  std::string mtx;
  std::string col;

  Scalar tau;              // regularization
  Index rank;              // rank
  Index lu_iter;           // LU iteration for SVD
  Index embedding_dim;     // final embedding dimensionality
  Index embedding_epochs;  // number of iterations

  bool kmeanspp;      // initialization by kmeans++
  Index knn;          // number of nearest neighbors
  Scalar knn_cutoff;  // cosine distance cutoff
  Index levels;       // number of eps levels

  bool out_data;  // output clustering data

  method_t method;
  bool prune_knn;

  std::string row_weight_file;

  bool verbose;
  Index min_size;

  bool raw_scale;
  bool log_scale;
  Scalar col_norm;

  void set_method(const std::string _method) {
    for (int j = 0; j < METHOD_NAMES.size(); ++j) {
      if (METHOD_NAMES.at(j) == _method) {
        method = static_cast<method_t>(j);
        TLOG("Use this clustering method: " << _method);
        break;
      }
    }
  }

  Index nystrom_sample;
  Index nystrom_batch;

  sampling_method_t nystrom_sample_method;

  void set_sampling_method(const std::string _method) {
    for (int j = 0; j < METHOD_NAMES.size(); ++j) {
      if (METHOD_NAMES.at(j) == _method) {
        nystrom_sample_method = static_cast<sampling_method_t>(j);
        break;
      }
    }
  }
};

int
parse_cluster_options(const int argc, const char* argv[],
                      cluster_options_t& options);

template <typename F0, typename F>
inline std::tuple<Mat, Mat, std::vector<Scalar> >
estimate_mixture_of_columns(const Mat& X, const cluster_options_t& options);

struct num_clust_t : public check_positive_t<Index> {
  explicit num_clust_t(const Index n) : check_positive_t<Index>(n) {}
};
struct num_sample_t : public check_positive_t<Index> {
  explicit num_sample_t(const Index n) : check_positive_t<Index>(n) {}
};

inline void
estimate_dbscan_of_columns(const Mat& X,                                  //
                           std::vector<std::vector<Index> >& membership,  //
                           const cluster_options_t& options);

inline Mat
embed_by_centroid(const Mat& X,                          // data matrix
                  const std::vector<Index>& membership,  // membership
                  const cluster_options_t& options);

inline std::vector<Index>
random_membership(const num_clust_t num_clust,  //
                  const num_sample_t num_sample);

/////////////////////
// implementations //
/////////////////////

template <typename F0, typename F>
inline Scalar
sum_log_marginal(const Mat& X,                    //
                 const Index K,                   //
                 std::vector<Index>& membership,  //
                 const cluster_options_t& options) {

  typename F0::dpm_alpha_t dpm_alpha(options.Alpha);
  typename F0::num_clust_t num_clust(K);
  using DS = discrete_sampler_t<Scalar, Index>;
  F0 prior(dpm_alpha, num_clust);

  const Index D = X.rows();
  const Index N = X.cols();
  typename F::dim_t dim(D);

  std::vector<Index> cindex(K);
  std::iota(cindex.begin(), cindex.end(), 0);
  std::vector<F> components;
  std::transform(cindex.begin(), cindex.end(), std::back_inserter(components),
                 [&dim](const Index) { return F(dim); });

  Scalar Ntot = 0.;

  for (Index i = 0; i < N; ++i) {
    const Index k = membership.at(i);
    if (k >= 0) {
      components[k] += X.col(i);
      Ntot += 1.0;
    }
  }

  Scalar ret     = 0.;
  auto _log_marg = [&ret, &components](const Index k) {
    ret += components.at(k).log_marginal();
  };
  std::for_each(cindex.begin(), cindex.end(), _log_marg);

  return (ret / Ntot);
}

template <typename T>
inline std::vector<T>
count_frequency(std::vector<T>& _membership, const T cutoff = 0) {

  const auto N = _membership.size();

  const T kk = *std::max_element(_membership.begin(), _membership.end()) + 1;

  std::vector<T> _sz(kk, 0);

  if (kk < 1) {
    return _sz;
  }

  for (std::size_t j = 0; j < N; ++j) {
    const T k = _membership.at(j);
    if (k >= 0) _sz[k]++;
  }

  return _sz;
}

template <typename T>
inline T
sort_cluster_index(std::vector<T>& _membership, const T cutoff = 0) {

  const auto N       = _membership.size();
  std::vector<T> _sz = count_frequency(_membership, cutoff);
  const T kk         = _sz.size();
  auto _order        = std_argsort(_sz);

  std::vector<T> rename(kk, -1);
  T k_new = 0;
  for (T k : _order) {
    if (_sz.at(k) >= cutoff) rename[k] = k_new++;
  }

  for (std::size_t j = 0; j < N; ++j) {
    const T k_old  = _membership.at(j);
    const T k_new  = rename.at(k_old);
    _membership[j] = k_new;
  }

  return k_new;
}

inline Mat
embed_by_centroid(const Mat& X,                    //
                  const std::vector<Index>& memb,  //
                  const cluster_options_t& options) {

  const Index kk = *std::max_element(memb.begin(), memb.end()) + 1;
  const Index D  = X.rows();
  const Index N  = X.cols();

  if (kk < 1) {
    TLOG("There is no centroid");
    return Mat{};
  }

  const Index n_rand_clust = 3;  // extra clusters for randomness

  Mat _cc(D, kk + n_rand_clust);  // take the means
  Vec _nn(kk + n_rand_clust);     // denominator
  _cc.setZero();
  _nn.setOnes();

  for (Index j = 0; j < N; ++j) {
    const Index k = memb.at(j);  //
    if (k >= 0) {                // ignore unassigned
      _cc.col(k) += X.col(j);    // accumulate
      _nn(k) += 1.0;             //
    }
  }

  /////////////////////////////
  // create random clusters  //
  /////////////////////////////

  std::random_device rd;
  std::mt19937 rgen(rd());

  std::vector<Index> rand_i(N);
  std::iota(rand_i.begin(), rand_i.end(), 0);
  std::shuffle(rand_i.begin(), rand_i.end(), rgen);

  const Index n_rand = (1 + N / kk) * n_rand_clust;

  if (options.verbose) {
    TLOG("Generating some cluster of random points");
  }

  for (Index r = 0; r < n_rand; ++r) {
    const Index j = r % N;
    const Index k = kk + (r % n_rand_clust);
    _cc.col(k) += X.col(j);
    _nn(k) += 1.0;
  }

  ////////////////////////////
  // normalize the centroid //
  ////////////////////////////

  Mat C = _cc * (_nn.cwiseInverse().asDiagonal());  // D x K

  // We might add stochastic noise
  auto _sigmoid = [](const Scalar x) -> Scalar {
    Scalar ret = 0.99 / (1.0 + exp(-x)) + 0.01;
    return ret;
  };

  const Index nepochs = options.embedding_epochs;
  const Index dd      = std::min(options.embedding_dim, kk);
  const Index rank    = dd + 1;

  if (options.verbose) {
    TLOG("Start embedding, epochs = " << nepochs << ", rank = " << rank);
  }

  Mat CC = scale_by_degree(C, options.tau);
  Mat XX = scale_by_degree(X, options.tau);

  ///////////////////////////
  // shatter the centroids //
  ///////////////////////////

  for (Index k = 0; k < CC.cols(); ++k) {
    CC.col(k) /= (CC.col(k).norm() + 1e-8);
  }

  Vec grad_D(D);

  for (Index t = 0; t < nepochs; ++t) {
    const Scalar rate =
        1.0 - static_cast<Scalar>(t) / static_cast<Scalar>(nepochs);

    for (Index k = 0; k < kk; ++k) {  // don't move the random one

      grad_D.setZero();
      for (Index l = 0; l < (kk + 1); ++l) {
        if (k == l) continue;
        const Scalar qq = _sigmoid(CC.col(l).dot(CC.col(k)));
        grad_D += CC.col(l) * qq;
      }

      CC.col(k) += rate * grad_D / static_cast<Scalar>(kk);
    }
  }

  // Initial embedding of X --> [rank x k] and [rank x n]
  RandomizedSVD<Mat> svd(rank, options.lu_iter);
  if (options.verbose) {
    svd.set_verbose();
  }
  svd.compute(CC);

  Mat Cd = svd.matrixV().rightCols(dd);
  Cd.transposeInPlace();

  Mat uu = svd.matrixU().rightCols(dd);
  Mat Xd = uu.transpose() * XX;

  if (options.verbose) {
    TLOG("Built initial embedding: " << Xd.rows() << " x " << Xd.cols());
  }

  if (kk < 2) {
    Xd.transposeInPlace();
    return Xd;
  }

  //////////////////////
  // Move data points //
  //////////////////////

  Vec grad_rank(dd);
  Vec pr(kk + 1);

  for (Index t = 0; t < nepochs; ++t) {

    const Scalar rate =
        1.0 - static_cast<Scalar>(t) / static_cast<Scalar>(nepochs);

    for (Index j = 0; j < N; ++j) {
      Index k = memb.at(j);
      if (k < 0) {
        pr = Cd.transpose() * Xd.col(j);
        pr.maxCoeff(&k);
      }

      grad_rank.setZero();
      const Scalar pp = _sigmoid(-Cd.col(k).dot(Xd.col(j)));
      grad_rank -= Cd.col(k) * pp * static_cast<Scalar>((kk - 1));

      for (Index l = 0; l < (kk + 1); ++l) {
        if (l == k) continue;
        const Scalar qq = _sigmoid(Cd.col(l).dot(Xd.col(j)));
        grad_rank += Cd.col(l) * qq;
      }

      Xd.col(j) += rate * grad_rank / static_cast<Scalar>(kk);
    }
  }

  if (options.verbose) {
    TLOG("Done the refinement: [" << Xd.rows() << " x " << Xd.cols() << "]");
  }
  Xd.transposeInPlace();
  return Xd;
}

template <typename T, typename OFS>
void
print_histogram(const std::vector<T>& nn,  //
                OFS& ofs,                  //
                const T height = 50.0,     //
                const T cutoff = .01,      //
                const int ntop = 10) {

  using std::accumulate;
  using std::ceil;
  using std::floor;
  using std::round;
  using std::setw;

  const Scalar ntot = (nn.size() <= ntop)
                          ? (accumulate(nn.begin(), nn.end(), 1e-8))
                          : (accumulate(nn.begin(), nn.begin() + ntop, 1e-8));

  ofs << "<histogram>" << std::endl;

  auto _print = [&](const Index j) {
    const Scalar x = nn.at(j);
    ofs << setw(10) << (j) << " [" << setw(10) << round(x) << "] ";
    for (int i = 0; i < ceil(x / ntot * height); ++i) ofs << "*";
    ofs << std::endl;
  };

  auto _args = std_argsort(nn);

  if (_args.size() <= ntop) {
    std::for_each(_args.begin(), _args.end(), _print);
  } else {
    std::for_each(_args.begin(), _args.begin() + ntop, _print);
  }
  ofs << "</histogram>" << std::endl;
}

inline void
estimate_dbscan_of_columns(const Mat& X,                                  //
                           std::vector<std::vector<Index> >& membership,  //
                           const cluster_options_t& options) {

  //////////////////////////////////////
  // Overall algorithm:		      //
  // 1. Construct kNN graph	      //
  // 2. Prune edges and vertices      //
  // 3. Identify connected components //
  //////////////////////////////////////

  const Index N = X.cols();
  const Index D = X.rows();

  TLOG("Constructing kNN graph ... N=" << N << ", knn=" << options.knn);

  const Index nnlist = std::max(options.knn + 1, options.nlist);
  const Index bilink =
      std::max(std::min(options.bilink, X.rows() - 1), static_cast<Index>(2));

  std::vector<std::tuple<Index, Index, Scalar> > knn_index;
  auto _knn = search_knn(SrcDataT(X.data(), X.rows(), X.cols()),  //
                         TgtDataT(X.data(), X.rows(), X.cols()),  //
                         KNN(options.knn + 1),                    // itself
                         BILINK(bilink),                          //
                         NNLIST(nnlist),                          //
                         knn_index);

  if (_knn != EXIT_SUCCESS) {
    TLOG("Failed to search k-nearest neighbors");
    return;
  }

  auto mutual_knn_index = prune_mutual_knn(knn_index);
  TLOG("Found " << mutual_knn_index.size() << " mutual kNN edges");

  membership.clear();

  for (Index l = 1; l <= options.levels; ++l) {

    const Scalar rr =
        static_cast<Scalar>(l) / static_cast<Scalar>(options.levels);

    const Scalar cutoff = options.knn_cutoff * rr;

    UGraph G;
    build_boost_graph(mutual_knn_index, G, cutoff);
    std::vector<Index> _membership(N);
    boost::connected_components(G, &_membership[0]);
    const Index kk = sort_cluster_index(_membership, options.min_size);
    membership.push_back(_membership);

    TLOG("K = " << kk << " components with distance <= " << cutoff);
    if (options.verbose) {
      auto nn = count_frequency(_membership, options.min_size);
      print_histogram(nn, std::cout);
      std::cout << std::flush;
    }
  }
}

template <typename F0, typename F>
inline std::tuple<Mat, Mat, std::vector<Scalar> >
estimate_mixture_of_columns(const Mat& X, const cluster_options_t& options) {

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

  std::vector<Scalar> elbo;
  elbo.reserve(2 + options.burnin_iter + options.max_iter);
  Vec mass(K);

  std::vector<Index> membership =
      random_membership(num_clust_t(K), num_sample_t(N));

  if (options.kmeanspp) {

    ////////////////////////////////////////////////////////
    // Kmeans++ initialization (Arthur and Vassilvitskii) //
    ////////////////////////////////////////////////////////

    {
      Scalar _elbo = 0;
      for (Index i = 0; i < N; ++i) {
        const Index k = membership.at(i);
        _elbo += components[k].elbo(X.col(i));
      }
      _elbo /= static_cast<Scalar>(N * D);
      TLOG("baseline[" << std::setw(5) << 0 << "] [" << std::setw(10) << _elbo
                       << "]");
      elbo.push_back(_elbo);
    }

    std::fill(membership.begin(), membership.end(), -1);
    {
      Vec x(D);
      x.setZero();
      DS sampler_n(N);
      Vec dist(N);

      for (Index k = 0; k < K; ++k) {

        dist = (X.colwise() - x)
                   .cwiseProduct(X.colwise() - x)
                   .colwise()
                   .sum()
                   .transpose();

        dist = dist.unaryExpr([](const Scalar _x) {
          return static_cast<Scalar>(0.5) * fasterlog(_x + 1e-8);
        });

        const Index j = sampler_n(dist);

        x = X.col(j).eval();
        components[k] += x;
        membership[j] = k;
        prior.add_to(k);
      }
      TLOG("Finished kmeans++ seeding");
    }

    {
      Scalar _elbo = 0;
      for (Index i = 0; i < N; ++i) {
        if (membership.at(i) < 0) {
          mass.setZero();
          for (Index k = 0; k < K; ++k) {
            mass(k) += components.at(k).log_marginal_ratio(X.col(i));
          }
          const Index l = sampler_k(mass);
          membership[i] = l;
          prior.add_to(l);
          components[l] += X.col(i);
        }

        const Index k = membership.at(i);
        _elbo += components[k].elbo(X.col(i));
      }

      _elbo /= static_cast<Scalar>(N * D);
      TLOG("Greedy- [" << std::setw(5) << 0 << "] [" << std::setw(10) << _elbo
                       << "]");
      elbo.push_back(_elbo);
    }
  } else {
    for (Index i = 0; i < N; ++i) {
      const Index k = membership.at(i);
      components[k] += X.col(i);
      prior.add_to(k);
    }
    Scalar _elbo = 0;
    for (Index i = 0; i < N; ++i) {
      const Index k = membership.at(i);
      _elbo += components[k].elbo(X.col(i));
    }
    _elbo /= static_cast<Scalar>(N * D);
    TLOG("baseline[" << std::setw(5) << 0 << "] [" << std::setw(10) << _elbo
                     << "]");
    elbo.push_back(_elbo);
  }

  /////////////////////////////////
  // burn-in to initialize again //
  /////////////////////////////////
  {
    for (Index b = 0; b < options.burnin_iter; ++b) {

      Scalar _elbo = 0;

      for (Index i = 0; i < N; ++i) {
        Index k_old = membership.at(i);
        components[k_old] -= X.col(i);
        prior.subtract_from(k_old);

        mass = prior.log_lcvi();
        for (Index k = 0; k < K; ++k) {
          mass(k) += components.at(k).log_marginal_ratio(X.col(i));
        }

        Index k_new   = sampler_k(mass);
        membership[i] = k_new;

        prior.add_to(k_new);
        components[k_new] += X.col(i);

        const Index k = membership.at(i);
        _elbo += components[k].elbo(X.col(i));
      }

      _elbo /= static_cast<Scalar>(N * D);
      TLOG("Burn-in [" << std::setw(5) << (b + 1) << "] [" << std::setw(10)
                       << _elbo << "]");
      elbo.push_back(_elbo);
    }
  }

  /////////////////////////
  // sort cluster labels //
  /////////////////////////

  sort_cluster_index(membership);
  for (Index k = 0; k < K; ++k) {
    components.at(k).clear();
  }

  Mat Z(K, N);
  Z.setZero();
  prior.clear();
  for (Index i = 0; i < N; ++i) {
    const Index k = membership.at(i);
    components[k] += X.col(i);
    prior.add_to(k);
    Z(k, i) = 1.0;
  }

  ////////////////////////
  // variational update //
  ////////////////////////

  Vec z_i(K);

  for (Index t = 0; t < options.max_iter; ++t) {

    Scalar _elbo = 0;

    for (Index i = 0; i < N; ++i) {

      ////////////////
      // Local step //
      ////////////////

      mass = prior.elbo();
      for (Index k = 0; k < K; ++k) {
        mass(k) += components.at(k).elbo(X.col(i));
      }

      normalized_exp(mass, z_i);

      /////////////////
      // Global step //
      /////////////////

      for (Index k = 0; k < K; ++k) {
        const Scalar z_old = Z(k, i);
        const Scalar z_new = z_i(k);

        components[k].update(X.col(i), z_old, z_new);

        _elbo += z_new * components[k].elbo(X.col(i));
      }

      Z.col(i) = z_i;
    }

    prior.update(Z, 1.0);
    elbo.push_back(_elbo);

    if (t >= options.min_iter) {
      const Scalar diff = std::abs(elbo.at(t) - elbo.at(t - 1));
      if (diff < options.Tol) break;
    }

    _elbo /= static_cast<Scalar>(N * D);
    const Index tt = 1 + t + options.burnin_iter;
    TLOG("VB Iter [" << std::setw(5) << tt << "] [" << std::setw(10) << _elbo
                     << "]");
  }

  Mat C(D, K);
  for (Index k = 0; k < components.size(); ++k) {
    C.col(k) = components.at(k).posterior_mean();
  }

  return std::make_tuple(Z, C, elbo);
}

//////////////////////////////////////////////////////
// A data-simulation routine for debugging purposes //
//////////////////////////////////////////////////////

inline std::tuple<Mat, std::vector<Index>, Mat>
simulate_gaussian_mixture(const Index n   = 300,     // sample size
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
  X = (centroid * Z)
          .unaryExpr([&](const Scalar x) { return x + sd * rnorm(gen); })
          .eval();
  return std::make_tuple(X, membership, centroid);
}

inline std::vector<Index>
random_membership(const num_clust_t num_clust,  //
                  const num_sample_t num_sample) {

  std::random_device rd{};
  std::mt19937 gen{rd()};
  const Index k = num_clust.val;
  const Index n = num_sample.val;

  std::uniform_int_distribution<Index> runifK{0, k - 1};
  std::vector<Index> idx(n);
  std::iota(idx.begin(), idx.end(), 0);
  std::vector<Index> ret;
  ret.reserve(n);

  std::transform(idx.begin(), idx.end(), std::back_inserter(ret),
                 [&](const Index i) { return runifK(gen); });

  return ret;
}

int
parse_cluster_options(const int argc,      //
                      const char* argv[],  //
                      cluster_options_t& options) {

  const char* _usage =
      "\n"
      "[Arguments]\n"
      "--data (-d)             : MTX file (data)\n"
      "--mtx (-d)              : MTX file (data)\n"
      "--sdata (-s)            : Spectral feature file (column x factor)\n"
      "--method (-M)           : A clustering method {DBSCAN,GMM}\n"
      "--col (-c)              : Column file\n"
      "--tau (-u)              : Regularization parameter (default: 1)\n"
      "--rank (-r)             : The maximal rank of SVD (default: 10)\n"
      "--lu_iter (-l)          : # of LU iterations (default: 3)\n"
      "--out (-o)              : Output file header (default: output)\n"
      "--row_weight (-w)       : Feature re-weighting (default: none)\n"
      "--col_norm (-C)         : Column normalization (default: 10000)\n"
      "--out_data (-D)         : Output clustering data (default: false)\n"
      "--log_scale (-L)        : Data in a log-scale (default: true)\n"
      "--raw_scale (-R)        : Data in a raw-scale (default: false)\n"
      "--nystrom_sample (-S)   : Nystrom sample size (default: 10000)\n"
      "--nystrom_batch (-B)    : Nystrom batch size (default: 10000)\n"
      "--sampling_method (-N)  : Nystrom sampling method: UNIFORM (default), "
      "CV, MEAN\n"
      "--verbose (-O)          : Output more words (default: false)\n"
      "\n"
      "[Options for DBSCAN]\n"
      "\n"
      "--knn (-k)              : K nearest neighbors (default: 10)\n"
      "--epsilon (-e)          : maximum cosine distance cutoff (default: "
      "1.0)\n"
      "--bilink (-m)           : # of bidirectional links (default: 10)\n"
      "--nlist (-f)            : # nearest neighbor lists (default: 10)\n"
      "--min_size (-z)         : minimum size to report (default: 10)\n"
      "--num_levels (-n)       : number of DBSCAN levels (default: 10)\n"
      "--prune_knn (-P)        : prune kNN graph (reciprocal match)\n"
      "--embedding_epochs (-t) : # of epochs for embedding\n"
      "\n"
      "[Options for Gaussian mixture models]\n"
      "\n"
      "--trunc (-K)            : maximum truncation-level of clustering\n"
      "--burnin (-I)           : burn-in (Gibbs) iterations (default: 10)\n"
      "--min_vbiter (-v)       : minimum VB iterations (default: 5)\n"
      "--max_vbiter (-V)       : maximum VB iterations (default: 100)\n"
      "--convergence (-T)      : epsilon value for checking convergence "
      "(default                : eps = 1e-8)\n"
      "--kmeanspp (-i)         : Kmeans++ initialization (default: false)\n"
      "\n"
      "[Details]\n"
      "Qin and Rohe (2013), Regularized Spectral Clustering under "
      "Degree-corrected Stochastic Block Model\n"
      "Li, Kwok, Lu (2010), Making Large-Scale Nystrom Approximation Possible\n"
      "\n"
      "[Details for kNN graph]\n"
      "\n"
      "(M)\n"
      "The number of bi-directional links created for every new element  \n"
      "during construction. Reasonable range for M is 2-100. Higher M work \n"
      "better on datasets with intrinsic dimensionality and/or high recall, \n"
      "while low M works better for datasets intrinsic dimensionality and/or\n"
      "low recalls. \n"
      "\n"
      "(N)\n"
      "The size of the dynamic list for the nearest neighbors (used during \n"
      "the search). A higher more accurate but slower search. This cannot be \n"
      "set lower than the number nearest neighbors k. The value ef of can be \n"
      "anything between of the dataset. [Reference] Malkov, Yu, and Yashunin. "
      "\n"
      "`Efficient and robust approximate nearest neighbor search using \n"
      "Hierarchical Navigable Small World graphs.` \n"
      "\n"
      "preprint: "
      "https://arxiv.org/abs/1603.09320 See also: "
      "https://github.com/nmslib/hnswlib";

  const char* const short_opts =
      "M:d:c:k:e:K:I:v:V:T:u:LR"
      "r:l:m:f:z:t:o:w:C:n:DPOih"
      "S:B:N:";

  const option long_opts[] = {
      {"mtx", required_argument, nullptr, 'd'},               //
      {"sdata", required_argument, nullptr, 's'},             //
      {"data", required_argument, nullptr, 'd'},              //
      {"sdata", required_argument, nullptr, 's'},             //
      {"method", required_argument, nullptr, 'M'},            //
      {"col", required_argument, nullptr, 'c'},               //
      {"knn", required_argument, nullptr, 'k'},               //
      {"epsilon", required_argument, nullptr, 'e'},           //
      {"trunc", required_argument, nullptr, 'K'},             //
      {"burnin", required_argument, nullptr, 'I'},            //
      {"min_vbiter", required_argument, nullptr, 'v'},        //
      {"max_vbiter", required_argument, nullptr, 'V'},        //
      {"convergence", required_argument, nullptr, 'T'},       //
      {"tau", required_argument, nullptr, 'u'},               //
      {"rank", required_argument, nullptr, 'r'},              //
      {"lu_iter", required_argument, nullptr, 'l'},           //
      {"bilink", required_argument, nullptr, 'm'},            //
      {"nlist", required_argument, nullptr, 'f'},             //
      {"out", required_argument, nullptr, 'o'},               //
      {"row_weight", required_argument, nullptr, 'w'},        //
      {"col_norm", required_argument, nullptr, 'C'},          //
      {"num_levels", required_argument, nullptr, 'n'},        //
      {"min_size", required_argument, nullptr, 'z'},          //
      {"embedding_epochs", required_argument, nullptr, 't'},  //
      {"out_data", no_argument, nullptr, 'D'},                //
      {"prune_knn", no_argument, nullptr, 'P'},               //
      {"verbose", no_argument, nullptr, 'O'},                 //
      {"log_scale", no_argument, nullptr, 'L'},               //
      {"raw_scale", no_argument, nullptr, 'R'},               //
      {"nystrom_sample", required_argument, nullptr, 'S'},    //
      {"nystrom_batch", required_argument, nullptr, 'B'},     //
      {"sampling_method", required_argument, nullptr, 'N'},   //
      {"kmeanspp", no_argument, nullptr, 'i'},                //
      {"help", no_argument, nullptr, 'h'},                    //
      {nullptr, no_argument, nullptr, 0}};

  while (true) {
    const auto opt = getopt_long(argc,                      //
                                 const_cast<char**>(argv),  //
                                 short_opts,                //
                                 long_opts,                 //
                                 nullptr);

    if (-1 == opt) break;

    switch (opt) {
      case 'd':
        options.mtx = std::string(optarg);
        break;
      case 's':
        options.spectral_file = std::string(optarg);
        break;
      case 'c':
        options.col = std::string(optarg);
        break;
      case 'k':
        options.knn = std::stoi(optarg);
        break;
      case 'e':
        options.knn_cutoff = std::stof(optarg);
        break;
      case 'K':
        options.K = std::stoi(optarg);
        break;
      case 'I':
        options.burnin_iter = std::stoi(optarg);
        break;
      case 'v':
        options.min_iter = std::stoi(optarg);
        break;
      case 'V':
        options.max_iter = std::stoi(optarg);
        break;
      case 'T':
        options.Tol = std::stof(optarg);
        break;
      case 'u':
        options.tau = std::stof(optarg);
        break;
      case 'r':
        options.rank = std::stoi(optarg);
        break;
      case 'l':
        options.lu_iter = std::stoi(optarg);
        break;
      case 'm':
        options.bilink = std::stoi(optarg);
        break;
      case 'n':
        options.levels = std::stoi(optarg);
        break;
      case 'f':
        options.nlist = std::stoi(optarg);
        break;
      case 'z':
        options.min_size = std::stoi(optarg);
        break;
      case 'o':
        options.out = std::string(optarg);
        break;
      case 'w':
        options.row_weight_file = std::string(optarg);
        break;
      case 'D':
        options.out_data = true;
        break;
      case 'P':
        options.prune_knn = true;
        break;
      case 'O':
        options.verbose = true;
        break;
      case 'M':
        options.set_method(std::string(optarg));
        break;
      case 'N':
        options.set_sampling_method(std::string(optarg));
        break;
      case 'i':
        options.kmeanspp = true;
        break;
      case 'S':
        options.nystrom_sample = std::stoi(optarg);
        break;
      case 'B':
        options.nystrom_batch = std::stoi(optarg);
        break;
      case 'L':
        options.log_scale = true;
        options.raw_scale = false;
        break;
      case 'R':
        options.log_scale = false;
        options.raw_scale = true;
        break;
      case 'C':
        options.col_norm = std::stof(optarg);
        break;
      case 't':
        options.embedding_epochs = std::stoi(optarg);
        break;
      case 'h':  // -h or --help
      case '?':  // Unrecognized option
        std::cerr << _usage << std::endl;
        return EXIT_FAILURE;
      default:  //
                ;
    }
  }

  return EXIT_SUCCESS;
}

#endif
