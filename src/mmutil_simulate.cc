#include "mmutil_simulate.hh"

int
main(const int argc, const char *argv[])
{
    simulate_options_t options;
    CHECK(parse_simulate_options(argc, argv, options));

    TLOG("Start simulating data...")
    CHECK(simulate_mtx_matrix(options));
    TLOG("Done");
    return EXIT_SUCCESS;
}
