#include "mmutil_cfa_col.hh"

int
main(const int argc, const char *argv[])
{

    cfa_options_t options;
    CHECK(parse_cfa_options(argc, argv, options));

    TLOG("CFA (Counter-Factual Adjustment)...")
    CHECK(cfa_col(options));
    TLOG("Done");

    return EXIT_SUCCESS;
}
