#include "mmutil_rand_col.hh"

void
print_help(const char *fname)
{
    const char *_desc =
        "[Arguments]\n"
        "Nsample: Number of uniformly randomly selected columns\n"
        "MTX:     Input matrix market file\n"
        "COLUMN:  Input column name file\n"
        "OUTPUT:  Output file header\n"
        "\n";

    std::cerr << "Randomly select columns" << std::endl;
    std::cerr << fname << " Nsample MTX COLUMN OUTPUT\n" << std::endl;
    std::cerr << _desc << std::endl;
}

int
main(const int argc, const char *argv[])
{
    if (argc < 5) {
        print_help(argv[0]);
        return EXIT_FAILURE;
    }

    using Str = std::string;
    const Index Nsample = std::stoi(argv[1]);

    const Str mtx_file(argv[2]);
    const Str column_file(argv[3]);
    const Str output(argv[4]);

    ERR_RET(!file_exists(mtx_file), "missing the mtx file");
    ERR_RET(!file_exists(column_file), "missing the column file");

    const Str output_col_file = output + ".columns.gz";
    const Str output_mtx_file = output + ".mtx.gz";

    copy_random_columns(Nsample,
                        mtx_file,
                        column_file,
                        output_mtx_file,
                        output_col_file);

    return EXIT_SUCCESS;
}
