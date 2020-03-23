#include "mmutil.hh"
#include "mmutil_cluster.hh"
#include "mmutil_match.hh"
#include "mmutil_spectral.hh"

#ifndef MMUTIL_SPECTRAL_CLUSTER_COL_HH
#define MMUTIL_SPECTRAL_CLUSTER_COL_HH

inline Mat
create_clustering_data(const cluster_options_t &options)
{
    using std::ignore;

    Vec weights;
    if (file_exists(options.row_weight_file)) {
        std::vector<Scalar> ww;
        CHECK(read_vector_file(options.row_weight_file, ww));
        weights = eigen_vector(ww);
    }

    svd_out_t svd = take_svd_online(options.mtx, weights, options);

    Mat Data = svd.V;        // sample x rank
    Data.transposeInPlace(); // rank x sample
    return Data;
}

template <typename Derived, typename S>
inline std::vector<std::tuple<S, Index>>
create_argmax_pair(const Eigen::MatrixBase<Derived> &Z,
                   const std::vector<S> &samples)
{
    ASSERT(Z.cols() == samples.size(),
           "#samples should correspond the columns of Z");

    auto _argmax = [&](const Index j) {
        Index ret;
        Z.col(j).maxCoeff(&ret);
        return std::make_tuple(samples.at(j), ret);
    };

    std::vector<Index> index(Z.cols());
    std::vector<std::tuple<S, Index>> membership;
    membership.reserve(Z.cols());
    std::iota(index.begin(), index.end(), 0);
    std::transform(index.begin(),
                   index.end(),
                   std::back_inserter(membership),
                   _argmax);

    return membership;
}

template <typename S>
inline std::vector<std::tuple<S, Index>>
create_argmax_pair(const std::vector<Index> &argmax,
                   const std::vector<S> &samples)
{
    const Index N = argmax.size();
    ASSERT(N == samples.size(), "#samples should correspond the columns of Z");

    auto _argmax = [&](const Index j) {
        const Index k = argmax.at(j);
        return std::make_tuple(samples.at(j), k);
    };

    std::vector<Index> index(N);
    std::vector<std::tuple<S, Index>> ret;
    ret.reserve(N);
    std::iota(index.begin(), index.end(), 0);
    std::transform(index.begin(),
                   index.end(),
                   std::back_inserter(ret),
                   _argmax);

    return ret;
}

template <typename Derived>
inline std::vector<Index>
create_argmax_vector(const Eigen::MatrixBase<Derived> &Z)
{
    const Index N = Z.cols();
    std::vector<Index> ret;
    ret.reserve(N);

    auto _argmax = [&Z](const Index j) {
        Index _ret;
        Z.col(j).maxCoeff(&_ret);
        return _ret;
    };

    std::vector<Index> index(N);
    std::iota(index.begin(), index.end(), 0);
    std::transform(index.begin(),
                   index.end(),
                   std::back_inserter(ret),
                   _argmax);

    return ret;
}

/**
   @param Data N x feature matrix
   @param options options
 */
void
run_mixture_model(const Mat &Data, const cluster_options_t &options)
{
    const Index N = Data.cols();

    using std::string;
    using std::vector;

    using F0 = trunc_dpm_t<Mat>;
    using F = multi_gaussian_component_t<Mat>;

    Mat Z, C;
    vector<Scalar> score;

    tie(Z, C, score) = estimate_mixture_of_columns<F0, F>(Data, options);

    ////////////////////////
    // output argmax file //
    ////////////////////////

    if (file_exists(options.col)) {
        vector<string> samples;
        CHECK(read_vector_file(options.col, samples));
        auto argmax = create_argmax_pair(Z, samples);
        write_tuple_file(options.out + ".argmax.gz", argmax);
    } else {
        vector<Index> samples(N);
        std::iota(samples.begin(), samples.end(), 0);
        auto argmax = create_argmax_pair(Z, samples);
        write_tuple_file(options.out + ".argmax.gz", argmax);
    }

    /////////////////////
    // show statistics //
    /////////////////////

    if (options.verbose) {
        Vec nn = Z * Mat::Ones(N, 1);
        vector<Scalar> count = std_vector(nn);
        print_histogram(count, std::cout);
        std::cout << std::flush;
    }

    write_data_file(options.out + ".centroid.gz", C);

    if (options.out_data) {
        write_data_file(options.out + ".data.gz", Data);
    }

    Z.transposeInPlace();
    write_data_file(options.out + ".clust_prob.gz", Z);

    write_vector_file(options.out + ".score.gz", score);

    TLOG("Done fitting a mixture model");
}

#endif
