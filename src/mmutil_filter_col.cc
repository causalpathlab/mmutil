#include "mmutil_filter_col.hh"

void
print_help(const char *fname)
{
    const char *_desc =
        "[Arguments]\n"
        "THRESHOLD: Minimum required number of non-zero elements in each column\n"
        "MTX:       Input matrix market file\n"
        "COLUMN:    Input column name file\n"
        "OUTPUT:    Output file header\n"
        "\n";

    std::cerr << "Filter in informative samples to reduce computational cost"
              << std::endl;
    std::cerr << std::endl;
    std::cerr << fname << " THRESHOLD MTX COLUMN OUTPUT" << std::endl;
    std::cerr << std::endl;
    std::cerr << _desc << std::endl;
}

//////////
// main //
//////////

int
main(const int argc, const char *argv[])
{
    if (argc < 5) {
        print_help(argv[0]);
        return EXIT_FAILURE;
    }

    using Str = std::string;
    const Index column_threshold = std::stoi(argv[1]);
    const Str mtx_file(argv[2]);
    const Str column_file(argv[3]);
    const Str output(argv[4]);

    ERR_RET(!file_exists(mtx_file), "missing the mtx file");
    ERR_RET(!file_exists(column_file), "missing the column file");

    filter_col_by_nnz(column_threshold, mtx_file, column_file, output);

    return EXIT_SUCCESS;
}
