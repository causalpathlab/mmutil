#include <getopt.h>
#include <unordered_map>

#include "mmutil.hh"
#include "mmutil_io.hh"
#include "mmutil_index.hh"
#include "utils/progress.hh"
#include "utils/std_util.hh"

#include "mmutil_pois.hh"

#ifndef MMUTIL_DIFF_HH_
#define MMUTIL_DIFF_HH_

struct diff_options_t {

    diff_options_t()
    {
        mtx_file = "";
        annot_prob_file = "";
        annot_file = "";
        annot_name_file = "";
        trt_ind_file = "";
        out = "output";
        verbose = false;

        gamma_a0 = 1;
        gamma_b0 = 1;

        discretize = true;
        normalize = false;
    }

    std::string mtx_file;
    std::string annot_prob_file;
    std::string annot_file;
    std::string col_file;
    std::string row_file;
    std::string trt_ind_file;
    std::string annot_name_file;
    std::string out;

    // For Bayesian calibration and Wald stat
    // Scalar wald_reg;
    Scalar gamma_a0;
    Scalar gamma_b0;

    bool verbose;

    // pois
    bool discretize;
    bool normalize;
};

struct diff_stat_collector_t {

    using scalar_t = Scalar;
    using index_t = Index;

    explicit diff_stat_collector_t(const std::vector<Index> &_conditions,
                                   const Index ncond,
                                   const Mat &_Z)
        : conditions(_conditions)
        , Ncond(ncond)
        , Z(_Z)
        , K(Z.cols())
    {
        max_row = 0;
        max_col = 0;
        max_elem = 0;
    }

    void eval_after_header(const Index r, const Index c, const Index e)
    {
        std::tie(max_row, max_col, max_elem) = std::make_tuple(r, c, e);

        ASSERT(max_col == conditions.size(),
               "the size of the condition vector should match with the data");

        ASSERT(Z.rows() == max_col, "rows(Z) must correspond to the columns");

        sum_mat.resize(max_row, K * Ncond);
        sum_mat.setZero();

        // #ifdef DEBUG
        // TLOG("Header: " << max_row << ", " << max_col << ", " << max_elem);
        // #endif
    }

    void eval(const Index row, const Index col, const Scalar w)
    {
        if (row < max_row && col < max_col) {
            const Index t = conditions[col];
            const Index g = row;

            for (Index k = 0; k < K; ++k) {
                const Scalar z_jk = Z(col, k);
                const Index j = pos(k, t);

                // if (j < 0 || j > K * Ncond) {
                // ELOG(j << ", " << k << ", " << t);
                // }

                sum_mat(g, j) += w * z_jk;
            }
        }
    }

    void eval_end_of_file()
    {
#ifdef DEBUG
        TLOG("Finished traversing");
#endif
    }

    inline Mat sum(const Index k) const
    {
        Mat ret(max_row, Ncond);
        ret.setZero();
        for (Index t = 0; t < Ncond; ++t)
            ret.col(t) = sum_mat.col(pos(k, t));
        return ret;
    }

    const std::vector<Index> &conditions; // condition for each column
    const Index Ncond;                    // # of conditions
    const Mat &Z;                         // annotation matrix
    const Index K;                        // # of annotations

private:
    Mat sum_mat; // max_row x (conditions * K)

    Index max_row;
    Index max_col;
    Index max_elem;

    inline Index pos(const Index k, const Index t) const
    {
        ASSERT(k >= 0 && k < K, "k out of range");
        ASSERT(t >= 0 && t < Ncond, "t out of range");
        return t * Ncond + k;
    }
};

template <typename OPTIONS>
int
test_diff(const OPTIONS &options)
{
    const std::string mtx_file = options.mtx_file;
    const std::string annot_prob_file = options.annot_prob_file;
    const std::string annot_file = options.annot_file;

    const std::string col_file = options.col_file;
    const std::string row_file = options.row_file;

    const std::string annot_name_file = options.annot_name_file;
    const std::string output = options.out;

    const std::string trt_file = options.trt_ind_file;

    const Scalar a0 = options.gamma_a0;
    const Scalar b0 = options.gamma_b0;

    //////////////////////////
    // column and row names //
    //////////////////////////

    std::vector<std::string> cols;
    CHECK(read_vector_file(col_file, cols));
    const Index Nsample = cols.size();

    std::vector<std::string> features;
    CHECK(read_vector_file(row_file, features));
    const Index Nfeature = features.size();

    /////////////////
    // label names //
    /////////////////

    std::vector<std::string> annot_name;
    CHECK(read_vector_file(annot_name_file, annot_name));
    auto lab_position = make_position_dict<std::string, Index>(annot_name);
    const Index K = annot_name.size();

    ///////////////////////
    // latent annotation //
    ///////////////////////

    Mat Z;

    if (annot_file.size() > 0) {
        Z.resize(Nsample, K);
        Z.setZero();

        std::unordered_map<std::string, std::string> annot_dict;
        CHECK(read_dict_file(annot_file, annot_dict));
        for (Index j = 0; j < cols.size(); ++j) {
            const std::string &s = cols.at(j);
            if (annot_dict.count(s) > 0) {
                const std::string &t = annot_dict.at(s);
                if (lab_position.count(t) > 0) {
                    const Index k = lab_position.at(t);
                    Z(j, k) = 1.;
                }
            }
        }
    } else if (annot_prob_file.size() > 0) {
        CHECK(read_data_file(annot_prob_file, Z));
    } else {
        return EXIT_FAILURE;
    }

    TLOG("Read latent annotations --> [" << Z.rows() << " x " << Z.cols()
                                         << "]");

    std::vector<std::string> trt_membership;

    CHECK(read_vector_file(trt_file, trt_membership));

    ASSERT(trt_membership.size() == Nsample,
           "Needs " << Nsample << " membership vector");

    TLOG("Reading condition vector --> [" << Nsample << " x 1]");

    std::vector<std::string> trt_id_name;
    std::vector<Index> trt_lab; // map: col -> trt index

    std::tie(trt_lab, trt_id_name, std::ignore) =
        make_indexed_vector<std::string, Index>(trt_membership);

    const Index T = trt_id_name.size();

    TLOG("Found " << T << " treatment conditions");

    diff_stat_collector_t collector(trt_lab, T, Z);
    visit_matrix_market_file(mtx_file, collector);

    TLOG("Collected sufficient statistics");

    // log-likelihood

    std::vector<std::tuple<std::string, std::string, Scalar, Scalar>> out_vec;

    out_vec.reserve(K * Nfeature);

    // lambda for each treatment condition
    Mat lambda_out(Nfeature * K, T);
    lambda_out.setZero();

    Mat sum_out(Nfeature * K, T);
    sum_out.setZero();

    Mat n_out(Nfeature * K, T);
    n_out.setZero();

    Mat lambda_null_out(Nfeature * K, 1);
    lambda_null_out.setZero();

    poisson_t::rate_opt_op_t rate_op(a0, b0);

    for (Index k = 0; k < K; ++k) {

        const Mat S1 = collector.sum(k);
        const Index D = S1.rows();

        if (features.size() != D) {
            ELOG("{# features in .rows} != {# features in .mtx}");
            return EXIT_FAILURE;
        }

        Mat N(D, T);
        N.setZero();

        for (Index j = 0; j < Z.rows(); ++j) {
            const Index t = trt_lab[j];
            N.col(t).array() += Z(j, k);
        }

        const Index ntot = N.sum();

        TLOG("Testing on " << ntot << " columns");

        const Mat lambda = S1.binaryExpr(N, rate_op);

        const Mat ln_lambda =
            lambda.unaryExpr([](const Scalar &x) { return fasterlog(x); });

        const Mat lambda_null =
            (S1 * Mat::Ones(T, 1)).binaryExpr(N * Mat::Ones(T, 1), rate_op);

        const Mat ln_lambda_null =
            lambda_null.unaryExpr([](const Scalar &x) { return fasterlog(x); });

        const Vec llik1 =
            (ln_lambda.cwiseProduct(S1) - lambda.cwiseProduct(N)) *
            Mat::Ones(T, 1);

        const Vec llik0 = ln_lambda_null.cwiseProduct(S1 * Mat::Ones(T, 1)) -
            lambda_null.cwiseProduct(N * Mat::Ones(T, 1));

        for (Index g = 0; g < D; ++g) {
            out_vec.emplace_back(std::make_tuple<>(features[g],
                                                   annot_name[k],
                                                   llik1(g),
                                                   llik0(g)));

            lambda_out.row(k * Nfeature + g) = lambda.row(g);
            lambda_null_out.row(k * Nfeature + g) = lambda_null.row(g);

            sum_out.row(k * Nfeature + g) = S1.row(g);
            n_out.row(k * Nfeature + g) = N.row(g);
        }
    }

    write_vector_file(output + ".cond.gz", trt_id_name);
    write_tuple_file(output + ".stat.gz", out_vec);
    write_data_file(output + ".lambda.gz", lambda_out);
    write_data_file(output + ".lambda_null.gz", lambda_null_out);
    write_data_file(output + ".sum.gz", sum_out);
    write_data_file(output + ".n.gz", n_out);

    return EXIT_SUCCESS;
}

template <typename OPTIONS>
int
parse_diff_options(const int argc,     //
                   const char *argv[], //
                   OPTIONS &options)
{
    const char *_usage =
        "\n"
        "[Arguments]\n"
        "--mtx (-m)        : data MTX file (M x N)\n"
        "--data (-m)       : data MTX file (M x N)\n"
        "--feature (-f)    : data row file (features)\n"
        "--row (-f)        : data row file (features)\n"
        "--col (-c)        : data column file (N x 1)\n"
        "--annot (-a)      : annotation/clustering assignment (N x 2)\n"
        "--annot_prob (-A) : annotation/clustering probability (N x K)\n"
        "--trt (-t)        : N x 1 sample to case-control membership\n"
        "--lab (-l)        : K x 1 annotation label name (e.g., cell type) \n"
        "--out (-o)        : Output file header\n"
        "\n"
        "--gamma_a0        : prior for gamma distribution(a0,b0) (default: 1)"
        "--gamma_b0        : prior for gamma distribution(a0,b0) (default: 1)"
        "\n";

    const char *const short_opts =
        "m:c:f:a:A:l:t:o:LRS:r:u:w:g:G:BDPC:k:b:n:hzv0:1:";

    const option long_opts[] = {
        { "mtx", required_argument, nullptr, 'm' },        //
        { "data", required_argument, nullptr, 'm' },       //
        { "annot_prob", required_argument, nullptr, 'A' }, //
        { "annot", required_argument, nullptr, 'a' },      //
        { "col", required_argument, nullptr, 'c' },        //
        { "row", required_argument, nullptr, 'f' },        //
        { "feature", required_argument, nullptr, 'f' },    //
        { "trt", required_argument, nullptr, 't' },        //
        { "trt_ind", required_argument, nullptr, 't' },    //
        { "lab", required_argument, nullptr, 'l' },        //
        { "label", required_argument, nullptr, 'l' },      //
        { "out", required_argument, nullptr, 'o' },        //
        { "discretize", no_argument, nullptr, 'D' },       //
        { "probabilistic", no_argument, nullptr, 'P' },    //
        { "a0", required_argument, nullptr, '0' },         //
        { "b0", required_argument, nullptr, '1' },         //
        { "gamma_a0", required_argument, nullptr, '0' },   //
        { "gamma_a1", required_argument, nullptr, '1' },   //
        { "verbose", no_argument, nullptr, 'v' },          //
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
        case 'm':
            options.mtx_file = std::string(optarg);
            break;
        case 'A':
            options.annot_prob_file = std::string(optarg);
            break;
        case 'a':
            options.annot_file = std::string(optarg);
            break;
        case 'c':
            options.col_file = std::string(optarg);
            break;
        case 'f':
            options.row_file = std::string(optarg);
            break;

        case 't':
            options.trt_ind_file = std::string(optarg);
            break;
        case 'l':
            options.annot_name_file = std::string(optarg);
            break;
        case 'o':
            options.out = std::string(optarg);
            break;

        case 'P':
            options.discretize = false;
            break;

        case 'D':
            options.discretize = true;
            break;

        case '0':
            options.gamma_a0 = std::stof(optarg);
            break;

        case '1':
            options.gamma_b0 = std::stof(optarg);
            break;

        case 'v': // -v or --verbose
            options.verbose = true;
            break;
        case 'h': // -h or --help
        case '?': // Unrecognized option
            std::cerr << _usage << std::endl;
            return EXIT_FAILURE;
        default: //
                 ;
        }
    }

    ERR_RET(!file_exists(options.mtx_file), "No MTX file");
    ERR_RET(!file_exists(options.annot_prob_file) &&
                !file_exists(options.annot_file),
            "No ANNOT or ANNOT_PROB file");
    ERR_RET(!file_exists(options.col_file), "No COL file");
    ERR_RET(!file_exists(options.row_file), "No ROW data file");
    ERR_RET(!file_exists(options.trt_ind_file), "No TRT file");
    ERR_RET(!file_exists(options.annot_name_file), "No LAB file");

    return EXIT_SUCCESS;
}

#endif
