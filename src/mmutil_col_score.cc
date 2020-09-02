#include "mmutil_score.hh"

void
print_help(const char *fname)
{
    const char *_desc = "[Arguments]\n"
                        "MTX:    Input matrix market file\n"
                        "COL:    Input col name file\n"
                        "OUTPUT: Output file name\n"
                        "\n";

    std::cerr << "Calculate scores for the columns in MTX" << std::endl;
    std::cerr << std::endl;
    std::cerr << fname << " MTX COL OUTPUT" << std::endl;
    std::cerr << std::endl;
    std::cerr << _desc << std::endl;
}

//////////
// main //
//////////

int
main(const int argc, const char *argv[])
{

    if (argc < 4) {
        print_help(argv[0]);
        return EXIT_FAILURE;
    }

    const std::string mtx_file(argv[1]);
    const std::string col_file(argv[2]);
    const std::string output(argv[3]);

    ERR_RET(!file_exists(mtx_file), "missing the mtx file");
    ERR_RET(!file_exists(col_file), "missing the col file");

    std::vector<std::string> col_names;
    CHECK(read_vector_file(col_file, col_names));

    Vec _cv, _sd, _mean;
    Index max_row, max_col;
    std::vector<Index> Nvec;
    std::tie(_mean, _sd, _cv, Nvec, max_row, max_col) =
        compute_mtx_col_stat(mtx_file);

    ASSERT(max_col == col_names.size(),
           "Found mismatch between the col names and .mtx file.");

    // name, n, mean, sd, cv
    using _tup = std::tuple<std::string, Index, Index, Scalar, Scalar, Scalar>;

    std::vector<_tup> out;
    out.reserve(col_names.size());

    for (Index r = 0; r < col_names.size(); ++r) {
        out.emplace_back(std::make_tuple(col_names.at(r),
                                         Nvec.at(r),
                                         max_row,
                                         _mean(r),
                                         _sd(r),
                                         _cv(r)));
    }

    write_tuple_file(output, out);

    return EXIT_SUCCESS;
}
