#include "mmutil_annotate_col.hh"

int
main(const int argc, const char *argv[])
{
    annotate_options_t options;

    CHECK(parse_annotate_options(argc, argv, options));

    return run_annotation(options);
}
