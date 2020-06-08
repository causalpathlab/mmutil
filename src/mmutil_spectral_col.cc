#include "mmutil.hh"
#include "mmutil_spectral.hh"
#include "mmutil_index.hh"
#include "mmutil_util.hh"
#include "mmutil_bgzf_util.hh"
#include "svd.hh"

int
main(const int argc, const char *argv[])
{
    spectral_options_t options;

    CHECK(parse_spectral_options(argc, argv, options));

    std::string mtx = options.mtx;
    std::string idx = options.idx;

    if (!is_file_bgz(mtx.c_str())) {
        mmutil::bgzf::convert_bgzip(mtx);
    }

    if (idx.length() == 0) {
        idx = mtx + ".index";
    }

    if (!file_exists(idx)) {
        TLOG("Creating indexes: " << idx);
        mmutil::index::build_mmutil_index(mtx, idx);
    }

    Mat ww;

    if (file_exists(options.row_weight_file)) {
        CHECK(read_data_file(options.row_weight_file, ww));
    }

    const std::string output = options.out;
    const std::string output_U_file = output + ".feature_factor.gz";
    const std::string output_V_file = output + ".sample_factor.gz";
    const std::string output_D_file = output + ".factor.gz";

    auto write_results = [&](const svd_out_t &svd) {
        write_data_file(output_U_file, svd.U);
        write_data_file(output_V_file, svd.V);
        write_data_file(output_D_file, svd.D);
        TLOG("Output results");
    };
    if (options.em_iter > 0) {
        svd_out_t svd = take_svd_online_em(mtx, idx, ww, options);
        write_results(svd);
    } else {
        svd_out_t svd = take_svd_online(mtx, idx, ww, options);
        write_results(svd);
    }
    TLOG("Done");

    return EXIT_SUCCESS;
}
