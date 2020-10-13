#include "mmutil_diff.hh"

int
main(const int argc, const char *argv[])
{

    diff_options_t options;
    CHECK(parse_diff_options(argc, argv, options));

    TLOG("Differential Rate Test");
    CHECK(test_diff(options));
    TLOG("Done");

    return EXIT_SUCCESS;
}
