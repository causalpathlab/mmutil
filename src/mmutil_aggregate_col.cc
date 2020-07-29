#include "mmutil_aggregate_col.hh"

int
main(const int argc, const char *argv[])
{

    aggregate_options_t options;
    CHECK(parse_aggregate_options(argc, argv, options));

    TLOG("Start aggregating...")
    CHECK(aggregate_col(options));
    TLOG("Done");

    return EXIT_SUCCESS;
}
