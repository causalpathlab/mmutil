#include "mmutil_aggregate_col.hh"

void
print_help(const char *fname)
{
    const char *_desc =
        "Aggregate columns to create summary stat matrices.\n"
        "\n"
        "[Arguments]\n"
        "MTX        : matrix market file (M x N)\n"
        "PROB       : N x K probability\n"
        "OUTPUT     : ${OUTPUT}.s1.gz ${OUTPUT}.s2.gz ${OUTPUT}.n.gz \n"
        "\n";
    std::cerr << fname << " MTX COL_MTX PROB ROW_PROB OUTPUT" << std::endl;
    std::cerr << std::endl;
    std::cerr << _desc << std::endl;
    std::cerr << std::endl;
}

int
main(const int argc, const char *argv[])
{
    if (argc != 4) {
        print_help(argv[0]);
        return EXIT_FAILURE;
    }

    const std::string mtx(argv[1]);
    const std::string prob(argv[2]);
    const std::string output(argv[3]);

    CHECK(aggregate_col(mtx, prob, output));

    TLOG("Done");
    return EXIT_SUCCESS;
}
