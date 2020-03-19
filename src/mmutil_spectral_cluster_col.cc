#include "mmutil_spectral_cluster_col.hh"

int
main(const int argc, const char *argv[])
{
    cluster_options_t options;
    CHECK(parse_cluster_options(argc, argv, options));

    const std::string mtx_file(options.mtx);
    const std::string output(options.out);

    Mat Data;

    if (file_exists(options.mtx)) {
        Data = create_clustering_data(options);
    } else if (file_exists(options.spectral_file)) {
        Mat uu;
        read_data_file(options.spectral_file, uu);
        Data = standardize(uu).transpose().eval();
    } else {
        TLOG("No input file exits. Try " << argv[0] << " -h");
        return EXIT_FAILURE;
    }

    // avoid too small or too large values
    const Scalar lb = -4, ub = 4;

    Data = Data.unaryExpr([&lb, &ub](const Scalar &x) -> Scalar {
        return std::min(std::max(x, lb), ub);
    });

    run_mixture_model(Data, options);
    return EXIT_SUCCESS;
}
