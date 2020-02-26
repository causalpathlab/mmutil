#include "mmutil_aggregate_col.hh"

void
print_help(const char *fname)
{
    const char *_desc =
        "Aggregate columns to create multiple types of summary stats.\n"
        "\n"
        "[Arguments]\n"
        "MTX        : Matrix Market file <i> <j> <value>\n"
        "COL        : Columns in the matrix <j>\n"
        "MEMBERSHIP : Membership probability <j> <k> <prob>\n"
        "OUTPUT     : ${OUTPUT}.s1.gz ${OUTPUT}.s2.gz ${OUTPUT}.n.gz \n"
        "\n";
    std::cerr << _desc << std::endl;
    std::cerr << fname << " MATCH MEMBERSHIP OUTPUT" << std::endl;
    std::cerr << std::endl;
}

int
main(const int argc, const char *argv[])
{
    if (argc != 5) {
        print_help(argv[0]);
        return EXIT_FAILURE;
    }

    const std::string mtx_file(argv[1]);
    const std::string col_file(argv[2]);
    const std::string membership_file(argv[3]);
    const std::string output(argv[4]);

    aggregate_col(mtx_file, col_file, membership_file, output);

    TLOG("Done");
    return EXIT_SUCCESS;
}
