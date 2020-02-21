#include <getopt.h>

#include <algorithm>
#include <random>
#include <unordered_map>
#include <unordered_set>

#include "eigen_util.hh"
#include "io.hh"
#include "mmutil.hh"
#include "mmutil_normalize.hh"
#include "mmutil_spectral.hh"
#include "mmutil_stat.hh"
#include "utils/progress.hh"

#ifndef MMUTIL_ANNOTATE_COL_
#define MMUTIL_ANNOTATE_COL_

struct annotate_options_t;

template <typename T>
std::tuple<SpMat, std::vector<std::string>, std::vector<std::string> >
read_annotation_matched(const T& options);

template <typename Derived, typename Derived2, typename T>
Mat
train_marker_genes(const Eigen::MatrixBase<Derived>& _L,
                   const Eigen::MatrixBase<Derived2>& _X, const T& options) {

  const Derived& L  = _L.derived();
  const Derived2& X = _X.derived();

  Mat xx = X;
  Mat mu = L;  // gene x label parameters

  mu.colwise().normalize();  // normalize for cosine similarity
  xx.colwise().normalize();  // normalize for cosine similarity

  Scalar score;

  Mat scoreMat(mu.cols(), xx.cols());  // label x cell
  Vec xj(mu.rows());
  Vec nsize(mu.cols());
  Vec mass(mu.cols());
  std::vector<Index> membership(xx.cols());
  std::fill(membership.begin(), membership.end(), 0);

  const Index max_em_iter = options.max_em_iter;
  Vec score_trace(max_em_iter);

  progress_bar_t<Index> prog(max_em_iter, 1);

  for (Index iter = 0; iter < max_em_iter; ++iter) {

    const Scalar t    = static_cast<Scalar>(iter + 1 + options.time_delay);
    const Scalar rate = std::pow(t, -options.rate_decay);

    /////////////////////////////////////////////////////
    // E-step: Identify best-matching labels for cells //
    /////////////////////////////////////////////////////

    scoreMat = mu.transpose() * xx;
    nsize.setZero();

    for (Index j = 0; j < scoreMat.cols(); ++j) {
      Index argmax;
      scoreMat.col(j).maxCoeff(&argmax);
      nsize(argmax) += 1.0;
      membership[j] = argmax;
    }

    //////////////////////////////////////////////////
    // M-step: Stochastic update for the parameters //
    //////////////////////////////////////////////////

    score = 0.0;

    for (Index j = 0; j < scoreMat.cols(); ++j) {
      Index argmax = membership.at(j);
      score += scoreMat(j, argmax);
      xj = xx.col(j).cwiseProduct(L.col(argmax));
      // mu.col(argmax) += rate * xj / nsize(argmax);
      mu.col(argmax) *= (1.0 - rate);
      mu.col(argmax) += rate * xj / nsize(argmax);
    }

    mu.colwise().normalize();

    score_trace(iter) = score;

    Scalar diff = std::abs(score_trace(iter - 1) - score_trace(iter));

    if (iter > 0) {
      diff /= (std::abs(score_trace(iter - 1)) + 1e-4);
    }

    prog.update();

    if (options.verbose) {
      prog(std::cerr);
    }

    if (iter >= 10 && diff < options.em_tol) {
      if (options.verbose) std::cerr << "\r" << std::endl;
      TLOG("Converged [" << std::setw(20) << diff << "]");
      break;
    }
  }

  if (options.verbose) std::cerr << "\r" << std::endl;

  return mu;
}

/////////////////////////////
// read matched annotation //
/////////////////////////////

template <typename T>
std::tuple<SpMat, std::vector<std::string>, std::vector<std::string> >
read_annotation_matched(const T& options) {

  using Str = std::string;

  std::unordered_map<Str, Str> pair_vec;
  read_pair_file<Str, Str>(options.ann, pair_vec);

  std::vector<Str> row_vec;
  read_vector_file(options.row, row_vec);

  std::unordered_map<Str, Index> row_pos;
  for (Index j = 0; j < row_vec.size(); ++j) {
    row_pos[row_vec.at(j)] = j;
  }

  std::unordered_map<Str, Index> label_pos;
  {
    Index j = 0;
    for (auto pp : pair_vec) {
      if (label_pos.count(std::get<1>(pp)) == 0)
        label_pos[std::get<1>(pp)] = j++;
    }
  }

  using ET = Eigen::Triplet<Scalar>;
  std::vector<ET> triples;

  for (auto pp : pair_vec) {
    if (row_pos.count(std::get<0>(pp)) > 0) {
      Index r = row_pos.at(std::get<0>(pp));
      Index l = label_pos.at(std::get<1>(pp));
      triples.push_back(ET(r, l, 1.0));
    }
  }

  SpMat L(row_pos.size(), label_pos.size());
  L.reserve(triples.size());
  L.setFromTriplets(triples.begin(), triples.end());

  std::vector<Str> labels(label_pos.size());
  std::vector<Str> rows(row_pos.size());

  for (auto pp : label_pos) {
    labels[std::get<1>(pp)] = std::get<0>(pp);
  }

  for (auto pp : row_pos) {
    rows[std::get<1>(pp)] = std::get<0>(pp);
  }

  return std::make_tuple(L, rows, labels);
}

//////////////////////
// Argument parsing //
//////////////////////

struct annotate_options_t {
  using Str = std::string;

  typedef enum { UNIFORM, CV, MEAN } sampling_method_t;
  const std::vector<Str> METHOD_NAMES;

  annotate_options_t() {
    mtx = "";
    col = "";
    row = "";
    ann = "";
    out = "output.txt.gz";

    col_norm = 10000;

    raw_scale = true;
    log_scale = false;

    initial_sample = 10000;
    batch_size     = 10000;

    max_em_iter = 100;
    time_delay  = 10;
    rate_decay  = 0.55;

    sampling_method = CV;

    em_tol = 1e-4;

    verbose = false;
  }

  Str mtx;
  Str col;
  Str row;
  Str ann;
  Str out;

  Scalar col_norm;

  bool raw_scale;
  bool log_scale;

  Index initial_sample;
  Index batch_size;

  Index max_em_iter;
  Scalar time_delay;  // (t + delay)^-discount
  Scalar rate_decay;  //

  sampling_method_t sampling_method;

  Scalar em_tol;

  void set_sampling_method(const std::string _method) {
    for (int j = 0; j < METHOD_NAMES.size(); ++j) {
      if (METHOD_NAMES.at(j) == _method) {
        sampling_method = static_cast<sampling_method_t>(j);
        break;
      }
    }
  }

  bool verbose;
};

template <typename T>
int
parse_annotate_options(const int argc,      //
                       const char* argv[],  //
                       T& options) {

  const char* _usage =
      "\n"
      "[Arguments]\n"
      "--mtx (-m)             : data MTX file\n"
      "--data (-m)            : data MTX file\n"
      "--col (-c)             : data column file\n"
      "--row (-r)             : data row file (features)\n"
      "--ann (-a)             : row annotation file\n"
      "--out (-o)             : Output file header\n"
      "\n"
      "--log_scale (-L)       : Data in a log-scale (default: false)\n"
      "--raw_scale (-R)       : Data in a raw-scale (default: true)\n"
      "\n"
      "--initial_sample (-I)  : Initial sample size (default: 10000)\n"
      "--batch_size (-B)      : Batch size (default: 10000)\n"
      "--sampling_method (-M) : Sampling method: CV (default), "
      "MEAN, UNIFORM\n"
      "\n"
      "--em_iter (-i)         : EM iteration (default: 100)\n"
      "--em_tol (-e)          : EM convergence criterion (default: 1e-4)\n"
      "--rate_decay (-y)      : Learning rate decay parameter (default: 0.55)\n"
      "--time_delay (-z)      : Delay for learning rate decay (default: 5)\n"
      "\n"
      "--verbose (-v)         : Set verbose (default: false)\n"
      "\n";

  const char* const short_opts = "m:d:c:r:a:o:I:B:M:LRi:e:y:z:hv";

  const option long_opts[] = {
      {"mtx", required_argument, nullptr, 'm'},              //
      {"data", required_argument, nullptr, 'm'},             //
      {"col", required_argument, nullptr, 'c'},              //
      {"row", required_argument, nullptr, 'r'},              //
      {"ann", required_argument, nullptr, 'a'},              //
      {"out", required_argument, nullptr, 'o'},              //
      {"log_scale", no_argument, nullptr, 'L'},              //
      {"raw_scale", no_argument, nullptr, 'R'},              //
      {"initial_sample", required_argument, nullptr, 'I'},   //
      {"batch_size", required_argument, nullptr, 'B'},       //
      {"sampling_method", required_argument, nullptr, 'M'},  //
      {"em_iter", required_argument, nullptr, 'i'},          //
      {"em_tol", required_argument, nullptr, 'e'},           //
      {"rate_decay", required_argument, nullptr, 'y'},       //
      {"time_delay", required_argument, nullptr, 'z'},       //
      {"help", no_argument, nullptr, 'h'},                   //
      {"verbose", no_argument, nullptr, 'v'},                //
      {nullptr, no_argument, nullptr, 0}};

  while (true) {
    const auto opt = getopt_long(argc,                      //
                                 const_cast<char**>(argv),  //
                                 short_opts,                //
                                 long_opts,                 //
                                 nullptr);

    if (-1 == opt) break;

    switch (opt) {
      case 'm':
        options.mtx = std::string(optarg);
        break;

      case 'c':
        options.col = std::string(optarg);
        break;

      case 'r':
        options.row = std::string(optarg);
        break;

      case 'a':
        options.ann = std::string(optarg);
        break;

      case 'o':
        options.out = std::string(optarg);
        break;

      case 'i':
        options.max_em_iter = std::stoi(optarg);
        break;

      case 'e':
        options.em_tol = std::stof(optarg);
        break;

      case 'z':
        options.time_delay = std::stoi(optarg);
        break;

      case 'y':
        options.rate_decay = std::stof(optarg);
        break;

      case 'I':
        options.initial_sample = std::stoi(optarg);
        break;
      case 'B':
        options.batch_size = std::stoi(optarg);
        break;
      case 'L':
        options.log_scale = true;
        options.raw_scale = false;
        break;
      case 'R':
        options.log_scale = false;
        options.raw_scale = true;
        break;
      case 'M':
        options.set_sampling_method(std::string(optarg));
        break;
      case 'v':  // -v or --verbose
        options.verbose = true;
        break;
      case 'h':  // -h or --help
      case '?':  // Unrecognized option
        std::cerr << _usage << std::endl;
        return EXIT_FAILURE;
      default:  //
                ;
    }
  }

  ERR_RET(!file_exists(options.mtx), "No MTX data file");
  ERR_RET(!file_exists(options.col), "No COL data file");
  ERR_RET(!file_exists(options.row), "No ROW data file");
  ERR_RET(!file_exists(options.ann), "No ANN data file");

  return EXIT_SUCCESS;
}

#endif
