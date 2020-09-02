#include "mmutil_filter_row.hh"

int
main(const int argc, const char *argv[])
{
    filter_row_options_t options;
    CHECK(parse_filter_row_options(argc, argv, options));
    return filter_row_by_score(options);
}
