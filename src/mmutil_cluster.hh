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
#include "utils/graph.hh"
#include "utils/progress.hh"

#ifndef MMUTIL_CLUSTER_HH_
#define MMUTIL_CLUSTER_HH_

struct cluster_options_t {

  typedef enum { DBSCAN, GAUSSIAN_MIXTURE } method_t;
  const std::vector<std::string> METHOD_NAMES;

  explicit cluster_options_t() : METHOD_NAMES{"DBSCAN", "GMM"} {

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
    knn           = 10;
    cosine_cutoff = 1e-2;
    levels        = 20;

    out      = "output";
    out_data = false;

    method = DBSCAN;

    tau     = 1.0;
    rank    = 10;
    lu_iter = 3;
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
  std::string mtx;
  std::string col;

  Scalar tau;     // regularization
  Index rank;     // rank
  Index lu_iter;  // LU iteration for SVD

  bool kmeanspp;         // initialization by kmeans++
  Index knn;             // number of nearest neighbors
  Scalar cosine_cutoff;  // cosine distance cutoff
  Index levels;          // number of eps levels

  bool out_data;  // output clustering data

  method_t method;

  void set_method(const std::string _method) {
    for (int j = 0; j < METHOD_NAMES.size(); ++j) {
      if (METHOD_NAMES.at(j) == _method) {
        method = static_cast<method_t>(j);
        TLOG("Use this clustering method: " << _method);
        break;
      }
    }
  }
};

int
parse_cluster_options(const int argc,      //
                      const char* argv[],  //
                      cluster_options_t& options) {

  const char* _usage =
      "\n"
      "[Arguments]\n"
      "--data (-d)        : MTX file (data)\n"
      "--mtx (-d)         : MTX file (data)\n"
      "--method (-M)      : A clustering method {DBSCAN,GMM}\n"
      "--col (-c)         : Column file\n"
      "--tau (-u)         : Regularization parameter (default: 1)\n"
      "--rank (-r)        : The maximal rank of SVD (default: 10)\n"
      "--luiter (-l)      : # of LU iterations (default: 3)\n"
      "--out (-o)         : Output file header (default: output)\n"
      "--out_data (-D)    : Output clustering data (default: false)\n"
      "\n"
      "[Options for DBSCAN]\n"
      "\n"
      "--knn (-k)         : K nearest neighbors (default: 10)\n"
      "--epsilon (-e)     : maximum cosine distance cutoff (default: 1e-2)\n"
      "--bilink (-m)      : # of bidirectional links (default: 10)\n"
      "--nlist (-f)       : # nearest neighbor lists (default: 10)\n"
      "\n"
      "[Options for Gaussian mixture models]\n"
      "\n"
      "--trunc (-K)       : maximum truncation-level of clustering\n"
      "--burnin (-B)      : burn-in (Gibbs) iterations (default: 10)\n"
      "--min_vbiter (-v)  : minimum VB iterations (default: 5)\n"
      "--max_vbiter (-V)  : maximum VB iterations (default: 100)\n"
      "--convergence (-T) : epsilon value for checking convergence (default: eps = 1e-8)\n"
      "--kmeanspp (-i)    : Kmeans++ initialization (default: false)\n"
      "\n"
      "[Details for kNN graph]\n"
      "\n"
      "(M)\n"
      "The number of bi-directional links created for every new element during construction.\n"
      "Reasonable range for M is 2-100. Higher M work better on datasets with high intrinsic\n"
      "dimensionality and/or high recall, while low M work better for datasets with low intrinsic\n"
      "dimensionality and/or low recalls.\n"
      "\n"
      "(N)\n"
      "The size of the dynamic list for the nearest neighbors (used during the search). A higher \n"
      "value leads to more accurate but slower search. This cannot be set lower than the number \n"
      "of queried nearest neighbors k. The value ef of can be anything between k and the size of \n"
      "the dataset.\n"
      "\n"
      "[Reference]\n"
      "Malkov, Yu, and Yashunin. `Efficient and robust approximate nearest neighbor search using\n"
      "Hierarchical Navigable Small World graphs.` preprint: https://arxiv.org/abs/1603.09320\n"
      "\n"
      "See also:\n"
      "https://github.com/nmslib/hnswlib\n"
      "\n";

  const char* const short_opts = "M:d:c:k:e:K:B:v:V:T:u:r:l:m:f:o:Dih";

  const option long_opts[] = {{"mtx", required_argument, nullptr, 'd'},          //
                              {"data", required_argument, nullptr, 'd'},         //
                              {"col", required_argument, nullptr, 'c'},          //
                              {"knn", required_argument, nullptr, 'k'},          //
                              {"epsilon", required_argument, nullptr, 'e'},      //
                              {"trunc", required_argument, nullptr, 'K'},        //
                              {"burnin", required_argument, nullptr, 'B'},       //
                              {"min_vbiter", required_argument, nullptr, 'v'},   //
                              {"max_vbiter", required_argument, nullptr, 'V'},   //
                              {"convergence", required_argument, nullptr, 'T'},  //
                              {"tau", required_argument, nullptr, 'u'},          //
                              {"rank", required_argument, nullptr, 'r'},         //
                              {"luiter", required_argument, nullptr, 'l'},       //
                              {"bilink", required_argument, nullptr, 'm'},       //
                              {"nlist", required_argument, nullptr, 'f'},        //
                              {"out", required_argument, nullptr, 'o'},          //
                              {"out_data", no_argument, nullptr, 'D'},           //
                              {"kmeanspp", no_argument, nullptr, 'i'},           //
                              {"help", no_argument, nullptr, 'h'},               //
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
      case 'c':
        options.col = std::string(optarg);
        break;
      case 'k':
        options.knn = std::stoi(optarg);
        break;
      case 'e':
        options.cosine_cutoff = std::stof(optarg);
        break;
      case 'K':
        options.K = std::stoi(optarg);
        break;
      case 'B':
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
      case 'f':
        options.nlist = std::stoi(optarg);
        break;
      case 'o':
        options.out = std::string(optarg);
        break;
      case 'D':
        options.out_data = true;
        break;
      case 'M':
        options.set_method(std::string(optarg));
        break;
      case 'i':
        options.kmeanspp = true;
        break;
      // case 't':
      //   options.repeat = std::stoi(optarg);
      //   break;
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

template <typename F0, typename F>
inline std::tuple<Mat, Mat, std::vector<Scalar> >
estimate_mixture_of_columns(const Mat& X, const cluster_options_t& options);

struct num_clust_t : public check_positive_t<Index> {
  explicit num_clust_t(const Index n) : check_positive_t<Index>(n) {}
};

struct num_sample_t : public check_positive_t<Index> {
  explicit num_sample_t(const Index n) : check_positive_t<Index>(n) {}
};

inline std::vector<Index>
random_membership(const num_clust_t num_clust,  //
                  const num_sample_t num_sample);

template <typename F0, typename F>
inline std::tuple<Mat, Mat, std::vector<Scalar> >
estimate_dbscan_of_columns(const Mat& X, const cluster_options_t& options);

////////////////////////////////////////////////////////////////

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

  for (Index i = 0; i < N; ++i) {
    const Index k = membership.at(i);
    components[k] += X.col(i);
  }

  Scalar ret     = 0.;
  auto _log_marg = [&ret, &components](const Index k) { ret += components.at(k).log_marginal(); };
  std::for_each(cindex.begin(), cindex.end(), _log_marg);

  return ret;
}

template <typename F0, typename F>
inline std::tuple<Mat, Mat, std::vector<Scalar> >
estimate_dbscan_of_columns(const Mat& X, const cluster_options_t& options) {

  // Overall algorithm:
  // 1. Construct kNN graph
  // 2. Prune edges and vertices
  // 3. Identify connected components

  const Index N = X.cols();
  const Index D = X.rows();

  TLOG("Constructing kNN graph ... N=" << N << ", knn=" << options.knn);

  Mat uu = X.colwise().normalized().eval();  // each col is data point

  const Index nnlist = std::max(options.knn + 1, options.nlist);
  const Index bilink = std::max(std::min(options.bilink, uu.rows() - 1), static_cast<Index>(2));

  std::vector<std::tuple<Index, Index, Scalar> > knn_index;
  auto _knn = search_knn(SrcDataT(uu.data(), uu.rows(), uu.cols()),  //
                         TgtDataT(uu.data(), uu.rows(), uu.cols()),  //
                         KNN(options.knn + 1),                       // itself
                         BILINK(bilink),                             //
                         NNLIST(nnlist),                             //
                         knn_index);

  if (_knn != EXIT_SUCCESS) {
    TLOG("Failed to search k-nearest neighbors");
    return std::make_tuple(Mat::Ones(1, N), Mat::Zero(D, 1), std::vector<Scalar>{});
  }

  const Scalar delta = options.cosine_cutoff / static_cast<Scalar>(options.levels);

  std::vector<Scalar> scores;
  scores.reserve(options.levels);
  std::vector<Index> Ncomponents;
  Ncomponents.reserve(options.levels);  

  Index K           = 1;
  Scalar best_score = 0.;
  std::vector< std::vector<Index> > membership;

  for (Index l = 1; l <= options.levels; ++l) {
    const Scalar cutoff = delta * l;
    UGraph G;
    build_boost_graph(knn_index, G, cutoff);
    std::vector<Index> _membership(N);
    const Index kk      = boost::connected_components(G, &_membership[0]);
    const Scalar _score = sum_log_marginal<F0, F>(X, kk, _membership, options);

    Ncomponents.push_back(kk);
    scores.push_back(_score);
    membership.push_back(_membership);

    if (l == 1 || _score > best_score) {
      best_score = _score;
      K          = kk;
    }

    TLOG("K = " << kk << " components with distance < " << cutoff << ", score = " << _score);
  }

  TLOG("Choosing K = " << K << " clusters");

  Mat C(D, K);
  Mat Z(K, N);
  Z.setZero();

  // for (Index j = 0; j < N; ++j) {
  //   Z(membership.at(j), j) += 1.0;
  // }

  // Vec nn(K);
  // nn.setConstant(1e-8);
  // nn += Z * Mat::Ones(N, 1);
  // C = X * Z.transpose() * nn.cwiseInverse().asDiagonal();

  TLOG("Done");

  return std::make_tuple(Z, C, scores);
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

  std::vector<Index> membership = random_membership(num_clust_t(K), num_sample_t(N));

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
      TLOG("baseline[" << std::setw(5) << 0 << "] [" << std::setw(10) << _elbo << "]");
      elbo.push_back(_elbo);
    }

    std::fill(membership.begin(), membership.end(), -1);
    {
      Vec x(D);
      x.setZero();
      DS sampler_n(N);
      Vec dist(N);

      for (Index k = 0; k < K; ++k) {
        dist = (X.colwise() - x).cwiseProduct(X.colwise() - x).colwise().sum().transpose();
        dist = dist.unaryExpr(
            [](const Scalar _x) { return static_cast<Scalar>(0.5) * fasterlog(_x + 1e-8); });
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
      TLOG("Greedy- [" << std::setw(5) << 0 << "] [" << std::setw(10) << _elbo << "]");
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
    TLOG("baseline[" << std::setw(5) << 0 << "] [" << std::setw(10) << _elbo << "]");
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
      TLOG("Burn-in [" << std::setw(5) << (b + 1) << "] [" << std::setw(10) << _elbo << "]");
      elbo.push_back(_elbo);
    }
  }

  ////////////////////////
  // variational update //
  ////////////////////////

  Mat Z(K, N);
  Z.setZero();
  for (Index i = 0; i < N; ++i) {
    const Index k = membership.at(i);
    Z(k, i)       = 1.0;
  }

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
    TLOG("VB Iter [" << std::setw(5) << tt << "] [" << std::setw(10) << _elbo << "]");
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
  X = (centroid * Z).unaryExpr([&](const Scalar x) { return x + sd * rnorm(gen); }).eval();
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

#endif
