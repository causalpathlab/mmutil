#include "mmutil_aggregate_col.hh"

int
main(const int argc, const char *argv[])
{

    aggregate_options_t options;
    CHECK(parse_aggregate_options(argc, argv, options));

    CHECK(aggregate_col(options.mtx,
                        options.mtx + ".index",
                        options.prob,
                        options.ind,
                        options.lab,
                        options.out,
                        options.batch_size,
                        options.log_scale));

    TLOG("Done");
    return EXIT_SUCCESS;
}
