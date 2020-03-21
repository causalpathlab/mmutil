#include <getopt.h>
#include <unordered_map>
#include "mmutil.hh"
#include "mmutil_stat.hh"
#include "mmutil_index.hh"
#include "mmutil_bgzf_util.hh"
#include "eigen_util.hh"
#include "io.hh"

#ifndef MMUTIL_AGGREGATE_COL_HH_
#define MMUTIL_AGGREGATE_COL_HH_

struct aggregate_options_t {
    using Str = std::string;

    typedef enum { UNIFORM, CV, MEAN } sampling_method_t;
    const std::vector<Str> METHOD_NAMES;

    aggregate_options_t()
    {
        mtx = "";
        prob = "";
        ind = "";
        lab = "";
        out = "output";
        batch_size = 10000;
        verbose = false;
    }

    Str mtx;
    Str prob;
    Str ind;
    Str lab;
    Str out;

    Index batch_size;
    bool verbose;
};

template <typename OPTIONS>
int
parse_aggregate_options(const int argc,     //
                        const char *argv[], //
                        OPTIONS &options)
{
    const char *_usage =
        "\n"
        "[Arguments]\n"
        "--mtx (-m)        : data MTX file (M x N)\n"
        "--data (-m)       : data MTX file (M x N)\n"
        "--prob (-p)       : annotation/clustering probability (N x K)\n"
        "--ind (-i)        : N x 1 sample to individual (n)\n"
        "--lab (-l)        : K x 1 label name\n"
        "--out (-o)        : Output file header\n"
        "\n"
        "--batch_size (-B) : Batch size (default: 10000)\n"
        "\n"
        "[Output]\n"
        "${output}_${lab}.s0.gz   : (M x n) sum 1 * z[j,k]\n"
        "${output}_${lab}.s1.gz   : (M x n) sum x[i,j] z[j,k]\n"
        "${output}_${lab}.s2.gz   : (M x n) sum x[i,j]^2 z[j,k]\n"
        "${output}_${lab}.cols.gz : (n x 1) name of the columns\n"
        "\n";

    const char *const short_opts = "m:p:i:l:o:B:hv";

    const option long_opts[] =
        { { "mtx", required_argument, nullptr, 'm' },        //
          { "data", required_argument, nullptr, 'm' },       //
          { "prob", required_argument, nullptr, 'p' },       //
          { "ind", required_argument, nullptr, 'i' },        //
          { "lab", required_argument, nullptr, 'l' },        //
          { "label", required_argument, nullptr, 'l' },      //
          { "out", required_argument, nullptr, 'o' },        //
          { "batch_size", required_argument, nullptr, 'B' }, //
          { "batchsize", required_argument, nullptr, 'B' },  //
          { "verbose", no_argument, nullptr, 'v' },          //
          { nullptr, no_argument, nullptr, 0 } };

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
            options.mtx = std::string(optarg);
            break;
        case 'p':
            options.prob = std::string(optarg);
            break;
        case 'i':
            options.ind = std::string(optarg);
            break;
        case 'l':
            options.lab = std::string(optarg);
            break;
        case 'o':
            options.out = std::string(optarg);
            break;
        case 'B':
            options.batch_size = std::stoi(optarg);
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

    ERR_RET(!file_exists(options.mtx), "No MTX data file");
    ERR_RET(!file_exists(options.prob), "No PROB data file");
    ERR_RET(!file_exists(options.ind), "No IND data file");
    ERR_RET(!file_exists(options.lab), "No LAB data file");

    return EXIT_SUCCESS;
}

int
aggregate_col(const std::string mtx_file,
              const std::string idx_file,
              const std::string prob_file,
              const std::string ind_file,
              const std::string lab_file,
              const std::string output,
              const Index batch_size = 30000)
{

    Mat Z;
    CHECK(read_data_file(prob_file, Z));
    TLOG("Latent membership matrix: " << Z.rows() << " x " << Z.cols());

    std::vector<std::string> ind;
    ind.reserve(Z.rows());
    CHECK(read_vector_file(ind_file, ind));

    ASSERT(ind.size() == Z.rows(),
           "Individual membership file mismatches with Z");

    std::unordered_map<std::string, Index> group_idx;
    std::vector<std::string> group_name;
    std::vector<Index> group;
    group.reserve(ind.size());

    for (Index i = 0; i < ind.size(); ++i) {
        const std::string &ii = ind.at(i);
        if (group_idx.count(ii) == 0) {
            const Index j = group_idx.size();
            group_idx[ii] = j;
            group_name.push_back(ii);
        }
        group.emplace_back(group_idx.at(ii));
    }

    const Index Nsample = ind.size();
    const Index Nind = group_idx.size();
    const Index K = Z.cols();

    TLOG("Identified " << Nind << " individuals");

    std::vector<std::string> lab;
    lab.reserve(K);
    CHECK(read_vector_file(lab_file, lab));

    ASSERT(lab.size() == K,
           "Need the same number of label names for the columns of Z");

    TLOG("Identified " << K << " labels");

    // Read an expression matrix block by block

    ASSERT(Z.rows() == Nsample, "rows(Z) != Nsample");
    const Scalar eps = 1e-8;

    // Indexing if needed
    CHECK(build_mmutil_index(mtx_file, idx_file));
    std::vector<idx_pair_t> idx_tab;
    CHECK(read_mmutil_index(idx_file, idx_tab));

    auto nz = [&eps](const Scalar &x) -> Scalar { return x < eps ? 0. : 1.0; };

    for (Index k = 0; k < K; ++k) {

        TLOG("Aggregating on " << lab.at(k) << "...");

        std::vector<Eigen::Triplet<Scalar>> triples;
        triples.reserve(Nsample);

        for (Index j = 0; j < Nsample; ++j) {
            const Scalar pr_jk = Z(j, k);
            if (pr_jk < eps)
                continue;
            triples.emplace_back(Eigen::Triplet<Scalar>(j, group.at(j), pr_jk));
        }

        SpMat Zk(Nsample, Nind);
        Zk.setFromTriplets(triples.begin(), triples.end());

        //////////////////////////////////////
        // collect statistics from the data //
        //////////////////////////////////////

        Mat S0, S1, S2;

        for (Index lb = 0; lb < Nsample; lb += batch_size) {
            const Index ub = std::min(Nsample, batch_size + lb);
            std::vector<Index> subcols_b(ub - lb);

            std::iota(subcols_b.begin(), subcols_b.end(), lb);
            TLOG("Reading data on the batch [" << lb << ", " << ub << ")");
            SpMat X_b =
                read_eigen_sparse_subset_col(mtx_file, idx_tab, subcols_b);
            SpMat Zk_b = row_sub(Zk, subcols_b);

            SpMat S0_b = X_b.unaryExpr(nz) * Zk_b;     //
            SpMat S1_b = X_b * Zk_b;                   // feature x Nind
            SpMat S2_b = X_b.cwiseProduct(X_b) * Zk_b; // feature x Nind

            if (lb == 0) {
                S0.resize(S0_b.rows(), Nind);
                S0.setZero();
                S1.resize(S1_b.rows(), Nind);
                S1.setZero();
                S2.resize(S2_b.rows(), Nind);
                S2.setZero();
            }

            S0 += S0_b;
            S1 += S1_b;
            S2 += S2_b;

            TLOG("S1 " << S1.rows() << " x " << S1.cols());
        }

        const std::string clust_name = lab.at(k);

        const std::string out_hdr = output + "_" + clust_name;

        write_vector_file(out_hdr + ".cols.gz", group_name);
        write_data_file(out_hdr + ".s0.gz", S0);
        write_data_file(out_hdr + ".s1.gz", S1);
        write_data_file(out_hdr + ".s2.gz", S2);

        TLOG("Wrote files for " << lab.at(k));
    }
    return EXIT_SUCCESS;
}

#endif
