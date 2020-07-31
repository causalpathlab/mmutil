#include "mmutil_bbknn.hh"

int
main(const int argc, const char *argv[])
{

    bbknn_options_t options;
    CHECK(parse_bbknn_options(argc, argv, options));

    TLOG("Start building BBKNN graph...")
    CHECK(build_bbknn(options));
    TLOG("Done");

    return EXIT_SUCCESS;
}
