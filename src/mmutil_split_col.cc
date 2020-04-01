#include "mmutil_split_col.hh"

void
print_help(const char *fname)
{
    const char *_desc =
        "Split columns into multiple file sets\n"
        "\n"
        "[Arguments]\n"
        "MTX        : Matrix Market file <i> <j> <value>\n"
        "MEMBERSHIP : Discrete membership <j> <k>\n"
        "OUTPUT     : ${OUTPUT}_${k}.mtx.gz, ${OUTPUT}_${k}.columns.gz\n"
        "\n";

    std::cerr << _desc << std::endl;
    std::cerr << fname << " MTX MEMBERSHIP OUTPUT" << std::endl;
    std::cerr << std::endl;
}

int
main(const int argc, const char *argv[])
{
    // if (argc < 5) {
    //     print_help(argv[0]);
    //     return EXIT_FAILURE;
    // }

    // using Str = std::string;

    // const Str mtx_file(argv[1]);
    // const Str membership_file(argv[2]);
    // const Str output(argv[3]);

    // ERR_RET(!file_exists(mtx_file), "missing the mtx file");
    // ERR_RET(!file_exists(membership_file), "missing the mtx file");

    // split_columns(mtx_file, membership_file, output);

    // TLOG("Done");
    return EXIT_SUCCESS;
}
