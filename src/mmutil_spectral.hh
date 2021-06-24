#include <getopt.h>

#include <random>

#include "mmutil.hh"
#include "mmutil_io.hh"
#include "mmutil_normalize.hh"
#include "mmutil_stat.hh"
#include "svd.hh"
#include "mmutil_index.hh"
#include "mmutil_bgzf_util.hh"
#include "mmutil_util.hh"
#include "ext/tabix/bgzf.h"

#ifndef MMUTIL_SPECTRAL_HH_
#define MMUTIL_SPECTRAL_HH_

struct spectral_options_t {
    using Str = std::string;

    typedef enum { UNIFORM, CV, MEAN } sampling_method_t;
    const std::vector<Str> METHOD_NAMES;

    spectral_options_t()
        : METHOD_NAMES { "UNIFORM", "CV", "MEAN" }
    {
        mtx = "";
        idx = "";
        out = "output.txt.gz";

        tau = 1.0;
        rank = 50;
        lu_iter = 5;
        col_norm = 10000;

        raw_scale = true;
        log_scale = false;
        row_weight_file = "";

        initial_sample = 10000;
        block_size = 10000;

        sampling_method = CV;

        rand_seed = 1;
        verbose = false;

        em_iter = 0;
        em_tol = 1e-2;
        em_recalibrate = true;

        check_index = false;
    }

    Str mtx;
    Str idx;
    Str out;

    Scalar tau;
    Index rank;
    Index lu_iter;
    Scalar col_norm;

    bool raw_scale;
    bool log_scale;
    bool verbose;
    Str row_weight_file;

    Index initial_sample;
    Index block_size;

    Index em_iter;
    Scalar em_tol;
    bool em_recalibrate;

    bool check_index;

    sampling_method_t sampling_method;

    int rand_seed;

    void set_sampling_method(const std::string _method)
    {
        for (int j = 0; j < METHOD_NAMES.size(); ++j) {
            if (METHOD_NAMES.at(j) == _method) {
                sampling_method = static_cast<sampling_method_t>(j);
                break;
            }
        }
    }
};

/**
   Batch-normalized graph Laplacian.
   - If needed, apply weights on features (rows; genes) to X matrix.

   @param X0 sparse data matrix
   @param weights row weights
   @param tau_scale regularization
   @param norm_target targetting normalization value
   @param log_trans do log(1+x) transformation

   Why is this a graph Laplacian?
   (1) We let adjacency matrix A = X'X assuming elements in X are non-negative
   (2) Let the Laplacian L = I - D^{-1/2} A D^{-1/2}
   = I - D^{-1/2} (X'X) D^{-1/2}
*/
template <typename Derived, typename Derived2>
inline Mat
make_normalized_laplacian(const Eigen::SparseMatrixBase<Derived> &_X0,
                          const Eigen::MatrixBase<Derived2> &_weights,
                          const float tau_scale,
                          const float norm_target = 0,
                          const bool log_trans = true)
{
    const Derived &X0 = _X0.derived();
    const Derived2 &weights = _weights.derived();
    const Index max_row = X0.rows();

    ASSERT(weights.rows() == max_row,
           "We need weights on each row. W: "
               << weights.rows() << " x " << weights.cols()
               << " vs. X: " << X0.rows() << " x " << X0.cols());
    ASSERT(weights.cols() == 1, "Provide summary vector");

    auto trans_fun = [&log_trans](const Scalar &x) -> Scalar {
        if (x < 0.0)
            return 0.;
        return log_trans ? fasterlog(x + 1.0) : x;
    };

    SpMat X(X0.rows(), X0.cols());

    if (norm_target > 0.) {
#ifdef DEBUG
        TLOG("Normalized to fixed value: " << norm_target);
#endif
        X = normalize_to_fixed(X0, norm_target).unaryExpr(trans_fun).eval();
    } else {
#ifdef DEBUG
        TLOG("Normalized to median");
#endif
        X = normalize_to_median(X0).unaryExpr(trans_fun).eval();
    }

#ifdef DEBUG
    TLOG("X: " << X.rows() << " x " << X.cols());
#endif

    ////////////////////////////////////////////////////////
    // make X(g,i) <- X(g,i) * min{1/sqrt(weight(g)),  1} //
    ////////////////////////////////////////////////////////

    auto _row_fun = [](const Scalar &x) -> Scalar {
        return x <= 0.0 ? 0.0 : std::sqrt(1.0 / x);
    };

    const Mat _rr = weights.unaryExpr(_row_fun);

#ifdef DEBUG
    TLOG("rows_denom: " << _rr.rows() << " x " << _rr.cols());
#endif

    //////////////////////////////////////////////
    // make X(g,i) <- X(g,i) / sqrt(D(i) + tau) //
    //////////////////////////////////////////////

    const Mat col_deg = X.cwiseProduct(X).transpose() * Mat::Ones(X.rows(), 1);
    const Scalar tau = col_deg.mean() * tau_scale;

    auto _col_fun = [&tau](const Scalar &x) -> Scalar {
        const Scalar _one = 1.0;
        return _one / std::sqrt(std::max(_one, x + tau));
    };

    const Mat _cc = col_deg.unaryExpr(_col_fun);

    ////////////////////
    // normalize them //
    ////////////////////

    Mat xx = _rr.asDiagonal() * X * _cc.asDiagonal();
    // Mat ret = standardize(xx); // why?
    return xx;
}

template <typename Derived, typename Derived2>
inline Mat
make_normalized_laplacian(const Eigen::MatrixBase<Derived> &_X0,
                          const Eigen::MatrixBase<Derived2> &_weights,
                          const float tau_scale,
                          const float norm_target,
                          const bool log_trans = true)
{

    Mat xx = _X0.derived();
    const Derived2 &ww = _weights.derived();

    normalize_columns(xx);
    xx *= norm_target;

    auto log_op = [](const Scalar &x) -> Scalar {
        return x >= 0. ? std::log(1.0 + x) : 0.;
    };

    if (log_trans)
        xx = xx.unaryExpr(log_op);

    const Mat _rr = ww.unaryExpr([](const Scalar &x) -> Scalar {
        return x <= 0.0 ? 0.0 : std::sqrt(1.0 / x);
    });

    const Mat col_deg =
        xx.cwiseProduct(xx).transpose() * Mat::Ones(xx.rows(), 1);

    const Scalar tau = col_deg.mean() * tau_scale;
    const Mat _cc = col_deg.unaryExpr([&tau](const Scalar &x) -> Scalar {
        const Scalar _one = 1.0;
        return _one / std::sqrt(std::max(_one, x + tau));
    });

    Mat ret = _rr.asDiagonal() * xx * _cc.asDiagonal();
    return ret;
}

/**
   @param mtx_file
   @param idx_file
   @param options
 */
template <typename OPTIONS>
std::tuple<SpMat, IntVec>
nystrom_sample_columns(const std::string mtx_file,
                       std::vector<Index> &idx_tab,
                       const OPTIONS &options)
{
    using namespace mmutil::io;
    TLOG("Collecting stats from the matrix file " << mtx_file);

    col_stat_collector_t collector;

    visit_matrix_market_file(mtx_file, collector);

    const Vec &s1 = collector.Col_S1;
    const Vec &s2 = collector.Col_S2;
    const IntVec &nnz_col = collector.Col_N;

    const Index N = collector.max_col;
    const Index nn = std::min(N, options.initial_sample);

    if (options.verbose)
        TLOG("Estimated statistics");

    std::random_device rd;
    std::mt19937 rgen(rd());

    std::vector<Index> index_r(N);

    if (options.sampling_method == OPTIONS::CV) {
        const Scalar nn = static_cast<Scalar>(collector.max_row);
        const Scalar mm = std::max(nn - 1.0, 1.0);

        auto cv_fun = [](const Scalar &v, const Scalar &m) -> Scalar {
            return std::sqrt(v) / (m + 1e-8);
        };

        Vec mu = s1 / nn;

        Vec score = ((s2 - s1.cwiseProduct(mu)) / mm).binaryExpr(mu, cv_fun);

        index_r = eigen_argsort_descending(score);

    } else if (options.sampling_method == OPTIONS::MEAN) {
        const Scalar n = static_cast<Scalar>(collector.max_row);

        Vec mu = s1 / n;
        index_r = eigen_argsort_descending(mu);

    } else {
        std::iota(index_r.begin(), index_r.end(), 0);
        std::shuffle(index_r.begin(), index_r.end(), rgen);
    }

    std::vector<Index> subcol(nn);
    std::copy(index_r.begin(), index_r.begin() + nn, subcol.begin());

    if (options.verbose)
        TLOG("Sampled " << nn << " columns");

    SpMat X =
        mmutil::io::read_eigen_sparse_subset_col(mtx_file, idx_tab, subcol);

    if (options.verbose)
        TLOG("Constructed sparse matrix: " << X.rows() << " x " << X.cols());

    return std::make_tuple(X, nnz_col);
}

struct svd_out_t {
    Mat U;
    Mat D;
    Mat V;
};

/**
   @param mtx_file
   @param idx_file
   @param weights
   @param options
 */
template <typename Derived, typename options_t>
inline svd_out_t
take_svd_online(const std::string mtx_file,
                const std::string idx_file,
                const Eigen::MatrixBase<Derived> &_weights,
                const options_t &options)
{

    const Scalar tau = options.tau;
    const Scalar norm = options.col_norm;
    const Index lu_iter = options.lu_iter;
    const Index block_size = options.block_size;
    const bool take_ln = options.log_scale;

    CHECK(mmutil::index::build_mmutil_index(mtx_file, idx_file));
    std::vector<Index> idx_tab;
    CHECK(mmutil::io::read_mmutil_index(idx_file, idx_tab));
    if (options.check_index)
        CHECK(mmutil::index::check_index_tab(mtx_file, idx_tab));

    //////////////////////////
    // step1 -- subsampling //
    //////////////////////////

    SpMat X;
    IntVec nnz_col;
    std::tie(X, nnz_col) = nystrom_sample_columns(mtx_file, idx_tab, options);
    const Index N = nnz_col.size();

    Vec ww(X.rows(), 1);
    ww.setOnes();

    if (_weights.size() > 0) {
        ww = _weights.derived();
        ASSERT(ww.rows() == X.rows(), "");
    }

    RandomizedSVD<Mat> svd(options.rank, lu_iter);
    {
        Mat xx =
            standardize(make_normalized_laplacian(X, ww, tau, norm, take_ln));
        svd.compute(xx);
    }

    Mat U = svd.matrixU();
    Mat Sig = svd.singularValues();

    const Index rank = U.cols();
    TLOG("Finished initial SVD: Effective number of factors = " << rank);

    //////////////////////////////////
    // step 2 -- Nystrom projection //
    //////////////////////////////////

    Mat proj = U * Sig.cwiseInverse().asDiagonal(); // feature x rank
    Mat Vt(rank, N);
    Vt.setZero();

    TLOG("Projection using the matrix: " << proj.rows() << " x "
                                         << proj.cols());

    Scalar err = 0;

    for (Index lb = 0; lb < N; lb += block_size) {

        const Index ub = std::min(N, block_size + lb);
        std::vector<Index> sub_b(ub - lb);
        std::iota(sub_b.begin(), sub_b.end(), lb);

        TLOG("Re-calibrating batch [" << lb << ", " << ub << ")");

        SpMat b =
            mmutil::io::read_eigen_sparse_subset_col(mtx_file, idx_tab, sub_b);

        Mat B =
            standardize(make_normalized_laplacian(b, ww, tau, norm, take_ln));

        for (Index i = 0; i < (ub - lb); ++i) {
            Vt.col(i + lb) += proj.transpose() * B.col(i);
        }

        err += (B - U * Sig.asDiagonal() * Vt.middleCols(lb, ub - lb))
                   .colwise()
                   .norm()
                   .sum();
    }

    err /= static_cast<Scalar>(N);
    TLOG("Finished Nystrom projection: " << err);

    Vt.transposeInPlace();
    return svd_out_t { U, Sig, Vt };
}

/**
   @param mtx_file
   @param idx_file
   @param weights
   @param options
 */
template <typename Derived, typename options_t>
inline svd_out_t
take_svd_online_em(const std::string mtx_file,
                   const std::string idx_file,
                   const Eigen::MatrixBase<Derived> &_weights,
                   const options_t &options)
{
    using namespace mmutil::io;
    const Scalar tau = options.tau;
    const Scalar norm = options.col_norm;
    const Index lu_iter = options.lu_iter;
    const bool take_ln = options.log_scale;

    CHECK(mmutil::bgzf::convert_bgzip(mtx_file));
    CHECK(mmutil::index::build_mmutil_index(mtx_file, idx_file));
    std::vector<Index> idx_tab;
    CHECK(mmutil::io::read_mmutil_index(idx_file, idx_tab));
    if (options.check_index)
        CHECK(mmutil::index::check_index_tab(mtx_file, idx_tab));

    mmutil::index::mm_info_reader_t info;
    CHECK(mmutil::bgzf::peek_bgzf_header(mtx_file, info));
    const Index numFeat = info.max_row;
    const Index N = info.max_col;

    Vec ww(numFeat, 1);
    ww.setOnes();

    if (_weights.size() > 0) {
        ww = _weights.derived();
        ASSERT(ww.rows() == numFeat,
               "The dim of weight vector differs from the data matrix: "
                   << ww.rows() << " vs. " << numFeat);
    }

    const Index rank = std::min(std::min(options.rank, numFeat), N);

    Mat U(numFeat, rank);
    Mat Sig(rank, 1);
    Mat Vt(rank, N);
    Vt.setZero();

    TLOG("SVD with rank = " << rank << " columns = " << N);

    const Index block_size = std::min(options.block_size, N);

    auto take_batch_data = [&](Index lb, Index ub) -> Mat {
        using namespace mmutil::index;
        std::vector<Index> sub_b(ub - lb);
        std::iota(sub_b.begin(), sub_b.end(), lb);
        SpMat x = read_eigen_sparse_subset_col(mtx_file, idx_tab, sub_b);

        return standardize(
            make_normalized_laplacian(x, ww, tau, norm, take_ln));
    };

#ifdef DEBUG
    TLOG("Initializing a dictionary matrix (U)");
#endif
    // Step 0. Initialize U matrix
    {
        std::random_device rd;
        std::mt19937 rgen(rd());
        std::vector<Index> index_r(N);
        std::iota(index_r.begin(), index_r.end(), 0);
        std::shuffle(index_r.begin(), index_r.end(), rgen);

        std::vector<Index> subcol(block_size);
        std::copy(index_r.begin(),
                  index_r.begin() + block_size,
                  subcol.begin());

        SpMat x =
            mmutil::io::read_eigen_sparse_subset_col(mtx_file, idx_tab, subcol);

        Mat xx = make_normalized_laplacian(x, ww, tau, norm, take_ln);
        Mat yy = standardize(xx);

#ifdef DEBUG
        TLOG("Training SVD");
#endif
        RandomizedSVD<Mat> svd(rank, lu_iter);
        if (options.verbose)
            svd.set_verbose();
        svd.compute(yy);
        U = svd.matrixU() * svd.singularValues().asDiagonal();
    }
#ifdef DEBUG
    std::cout << U.topRows(10) << std::endl;
#endif
    if (options.verbose)
        TLOG("Found initial U matrix");

    Mat XV(U.rows(), rank);
    Mat VtV(rank, rank);

    XV.setZero();
    VtV.setZero();

    const Scalar eps = 1e-8;
    auto safe_inverse = [&eps](const Scalar &x) -> Scalar {
        return 1.0 / (x + eps);
    };

    Eigen::JacobiSVD<Mat> svd_utu;
    Eigen::JacobiSVD<Mat> svd_vtv;

    Mat UtU(rank, rank);
    Mat UtUinv(rank, rank);
    Mat VtVinv(rank, rank);

    {
        // for the initial update of Vt
        UtU = U.transpose() * U;
        svd_utu.compute(UtU, Eigen::ComputeThinU | Eigen::ComputeThinV);
        UtUinv = svd_utu.matrixU() *
            (svd_utu.singularValues().unaryExpr(safe_inverse).asDiagonal()) *
            svd_utu.matrixV().transpose();
    }

    auto update_dictionary = [&]() {
        svd_vtv.compute(VtV, Eigen::ComputeThinU | Eigen::ComputeThinV);

        VtVinv = svd_vtv.matrixU() *
            (svd_vtv.singularValues().unaryExpr(safe_inverse).asDiagonal()) *
            svd_vtv.matrixV().transpose();

        U = XV * VtVinv;

        // for the update of Vt
        UtU = U.transpose() * U;
        svd_utu.compute(UtU, Eigen::ComputeThinU | Eigen::ComputeThinV);
        UtUinv = svd_utu.matrixU() *
            (svd_utu.singularValues().unaryExpr(safe_inverse).asDiagonal()) *
            svd_utu.matrixV().transpose();
    };

    Scalar err_prev = 0;
    Scalar tol = options.em_tol;

    for (Index t = 0; t < options.em_iter; ++t) {

#ifdef CPYTHON
        if (PyErr_CheckSignals() != 0) {
            ELOG("Interrupted at EM = " << (t + 1));
            break;
        }
#endif

        Scalar err_curr = 0;

        const Index nb = N / block_size + (N % block_size > 0 ? 1 : 0);

        for (Index lb = 0; lb < N; lb += block_size) {
            const Index ub = std::min(N, block_size + lb);

            Mat xx = take_batch_data(lb, ub);

            // discount the previous XV and VtV
            Mat vt = Vt.middleCols(lb, ub - lb);
            XV -= xx * vt.transpose();
            VtV -= vt * vt.transpose();

            // update with new v
            vt = UtUinv * U.transpose() * xx;

            Scalar nn = static_cast<Scalar>(xx.cols());
            XV += xx * vt.transpose();
            VtV += vt * vt.transpose();

            for (Index j = 0; j < vt.cols(); ++j)
                Vt.col(j + lb) = vt.col(j);

            Scalar _err = (xx - U * vt).colwise().norm().sum();

            if (options.verbose)
                TLOG("Batch [" << lb << ", " << ub << ") --> "
                               << _err / static_cast<Scalar>(xx.cols()));

            err_curr += _err;
        }

        err_curr /= static_cast<Scalar>(N);
        TLOG("Iter " << (t + 1) << " error = " << err_curr);

        if (std::abs(err_prev - err_curr) / (err_curr + 1e-8) < tol) {
            break;
        }
        err_prev = err_curr;
        update_dictionary();
    }

#ifdef DEBUG
    std::cout << U.topRows(10) << std::endl;
#endif

    // To ensure the orthogonality between columns
    Eigen::JacobiSVD<Mat> svd_u;
    svd_u.compute(U, Eigen::ComputeThinU | Eigen::ComputeThinV);
    U = svd_u.matrixU(); //
    Sig = svd_u.singularValues();
    Mat V = (svd_u.matrixV().transpose() * Vt).transpose();

    return svd_out_t { U, Sig, V };
}

/**
   @param mtx_file
   @param idx_file
   @param weights
   @param options
 */
template <typename Derived, typename options_t>
inline svd_out_t
take_svd_online_em(const std::string mtx_file,
                   const Eigen::MatrixBase<Derived> &_weights,
                   const options_t &options)
{
    const std::string idx_file = mtx_file + ".index";
    return take_svd_online_em(mtx_file, idx_file, _weights, options);
}

/**
   @param mtx_file
   @param weights
   @param options
 */
template <typename Derived, typename options_t>
inline svd_out_t
take_svd_online(const std::string mtx_file,
                const Eigen::MatrixBase<Derived> &_weights,
                const options_t &options)
{
    std::string idx_file = mtx_file + ".index";
    return take_svd_online(mtx_file, idx_file, _weights, options);
}

/**
   @param mtx_file data matrix file
   @param idx_file index file
   @param weights feature x 1
   @param proj feature x rank
   @param options
 */
template <typename Derived, typename Derived2, typename options_t>
inline Mat
take_proj_online(const std::string mtx_file,
                 const std::string idx_file,
                 const Eigen::MatrixBase<Derived> &_weights,
                 const Eigen::MatrixBase<Derived2> &_proj,
                 const options_t &options)
{

    CHECK(mmutil::index::build_mmutil_index(mtx_file, idx_file));
    std::vector<Index> idx_tab;
    CHECK(mmutil::io::read_mmutil_index(idx_file, idx_tab));

    mmutil::index::mm_info_reader_t info;

    CHECK(mmutil::bgzf::peek_bgzf_header(mtx_file, info));

    const Scalar tau = options.tau;
    const Scalar norm = options.col_norm;
    const Index block_size = options.block_size;
    const bool take_ln = options.log_scale;

    const Index D = info.max_row;
    const Index N = info.max_col;
    const Derived2 proj = _proj.derived();
    const Index rank = proj.cols();

    ASSERT(proj.rows() == info.max_row,
           "Projection matrix should have the same number of rows");

    Vec ww(D, 1);
    ww.setOnes();

    if (_weights.size() > 0) {
        ww = _weights.derived();
    }

    Mat V(N, rank);
    V.setZero();

    for (Index lb = 0; lb < N; lb += block_size) {

        const Index ub = std::min(N, block_size + lb);
        std::vector<Index> sub_b(ub - lb);
        std::iota(sub_b.begin(), sub_b.end(), lb);

        SpMat b =
            mmutil::io::read_eigen_sparse_subset_col(mtx_file, idx_tab, sub_b);
        Mat B = make_normalized_laplacian(b, ww, tau, norm, take_ln);
        B.transposeInPlace();

        for (Index i = 0; i < (ub - lb); ++i) {
            V.row(i + lb) += B.row(i) * proj;
        }
    }

    return V;
}

/**
   @param mtx_file data matrix file
   @param weights feature x 1
   @param proj feature x rank
   @param options
 */
template <typename Derived, typename Derived2, typename options_t>
inline Mat
take_proj_online(const std::string mtx_file,
                 const Eigen::MatrixBase<Derived> &_weights,
                 const Eigen::MatrixBase<Derived2> &_proj,
                 const options_t &options)
{
    return take_proj_online(mtx_file,
                            mtx_file + ".index",
                            _weights,
                            _proj,
                            options);
}

int
parse_spectral_options(const int argc,     //
                       const char *argv[], //
                       spectral_options_t &options)
{
    const char *_usage =
        "\n"
        "[Arguments]\n"
        "--data (-d)            : MTX file (data)\n"
        "--mtx (-d)             : MTX file (data)\n"
        "--tau (-u)             : Regularization parameter (default: tau = 1)\n"
        "--rank (-r)            : The maximal rank of SVD (default: rank = 50)\n"
        "--iter (-l)            : # of LU iterations (default: iter = 5)\n"
        "--row_weight (-w)      : Feature re-weighting (default: none)\n"
        "--col_norm (-C)        : Column normalization (default: 10000)\n"
        "--rand_seed (-s)       : Random seed (default: 1)\n"
        "--initial_sample (-S)  : Nystrom sample size (default: 10000)\n"
        "--block_size (-B)      : Nystrom batch size (default: 10000)\n"
        "--sampling_method (-M) : Nystrom sampling method: CV (default), "
        "UNIFORM, MEAN\n"
        "--log_scale (-L)       : Data in a log-scale (default: true)\n"
        "--raw_scale (-R)       : Data in a raw-scale (default: false)\n"
        "--out (-o)             : Output file name\n"
        "\n"
        "--verbose (-v)         : verbosity\n"
        "--em_iter (-i)         : EM iterations (default: 0)\n"
        "--em_tol (-t)          : EM convergence (default: 1e-2)\n"
        "\n"
        "--check_index          : check matrix market index (default: false)\n"
        "\n"
        "[Details]\n"
        "Qin and Rohe (2013), Regularized Spectral Clustering under "
        "Degree-corrected Stochastic Block Model\n"
        "Li, Kwok, Lu (2010), Making Large-Scale Nystrom Approximation Possible\n"
        "\n";

    const char *const short_opts = "d:m:u:r:l:C:w:S:s:B:ILRM:hvo:i:t:";

    const option long_opts[] = {
        { "mtx", required_argument, nullptr, 'd' },             //
        { "data", required_argument, nullptr, 'd' },            //
        { "out", required_argument, nullptr, 'o' },             //
        { "tau", required_argument, nullptr, 'u' },             //
        { "rank", required_argument, nullptr, 'r' },            //
        { "lu_iter", required_argument, nullptr, 'l' },         //
        { "row_weight", required_argument, nullptr, 'w' },      //
        { "col_norm", required_argument, nullptr, 'C' },        //
        { "log_scale", no_argument, nullptr, 'L' },             //
        { "raw_scale", no_argument, nullptr, 'R' },             //
        { "check_index", no_argument, nullptr, 'I' },           //
        { "rand_seed", required_argument, nullptr, 's' },       //
        { "initial_sample", required_argument, nullptr, 'S' },  //
        { "block_size", required_argument, nullptr, 'B' },      //
        { "sampling_method", required_argument, nullptr, 'M' }, //
        { "help", no_argument, nullptr, 'h' },                  //
        { "verbose", no_argument, nullptr, 'v' },               //
        { "em_iter", required_argument, nullptr, 'i' },         //
        { "em_tol", required_argument, nullptr, 't' },          //
        { nullptr, no_argument, nullptr, 0 }
    };

    while (true) {
        const auto opt = getopt_long(argc,                      //
                                     const_cast<char **>(argv), //
                                     short_opts,                //
                                     long_opts,                 //
                                     nullptr);

        if (-1 == opt)
            break;

        switch (opt) {
        case 'd':
            options.mtx = std::string(optarg);
            break;
        case 'o':
            options.out = std::string(optarg);
            break;
        case 'u':
            options.tau = std::stof(optarg);
            break;
        case 'C':
            options.col_norm = std::stof(optarg);
            break;
        case 'r':
            options.rank = std::stoi(optarg);
            break;
        case 'l':
            options.lu_iter = std::stoi(optarg);
            break;
        case 'w':
            options.row_weight_file = std::string(optarg);
            break;
        case 'S':
            options.initial_sample = std::stoi(optarg);
            break;
        case 's':
            options.rand_seed = std::stoi(optarg);
            break;
        case 'B':
            options.block_size = std::stoi(optarg);
            break;
        case 'i':
            options.em_iter = std::stoi(optarg);
            break;
        case 't':
            options.em_tol = std::stof(optarg);
            break;
        case 'I':
            options.check_index = true;
            break;
        case 'L':
            options.log_scale = true;
            options.raw_scale = false;
            break;
        case 'R':
            options.log_scale = false;
            options.raw_scale = true;
            break;
        case 'v':
            options.verbose = true;
            break;

        case 'M':
            options.set_sampling_method(std::string(optarg));
            break;

        case 'h': // -h or --help
        case '?': // Unrecognized option
            std::cerr << _usage << std::endl;
            return EXIT_FAILURE;
        default: //
                 ;
        }
    }

    return EXIT_SUCCESS;
}

#endif
