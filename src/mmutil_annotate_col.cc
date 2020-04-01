#include "mmutil_annotate_col.hh"

int
main(const int argc, const char *argv[])
{
    annotation_options_t options;

    CHECK(parse_annotation_options(argc, argv, options));

    return run_annotation(options);
}
