#include "mmutil_doublet_qc.hh"

int
main(const int argc, const char *argv[])
{
    doublet_qc_options_t options;
    CHECK(parse_doublet_qc_options(argc, argv, options));
    CHECK(run_doublet_qc(options));
}
