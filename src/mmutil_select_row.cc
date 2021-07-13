#include <algorithm>
#include <functional>
#include <string>
#include <unordered_set>

#include "mmutil_select.hh"
#include "mmutil_filter_col.hh"

void
print_help(const char *fname)
{
    const char *_desc = "[Arguments]\n"
                        "MTX:       [Input] Matrix market file\n"
                        "ROW:       [Input] Row name file\n"
                        "COL:       [Input] Column name file\n"
                        "SELECTED:  [Input] Row names to be selected\n"
                        "OUTPUT:    [Output] File header\n"
                        "\n";

    std::cerr << "Select a subset of rows" << std::endl;
    std::cerr << std::endl;
    std::cerr << fname << " MTX ROW COL SELECTED OUTPUT" << std::endl;
    std::cerr << std::endl;
    std::cerr << _desc << std::endl;
}

int
main(const int argc, const char *argv[])
{
    if (argc < 6) {
        print_help(argv[0]);
        return EXIT_FAILURE;
    }

    using Str = std::string;
    const Str mtx_file(argv[1]);
    const Str row_file(argv[2]);
    const Str col_file(argv[3]);
    const Str select_file(argv[4]);
    const Str output(argv[5]);

    ERR_RET(!file_exists(mtx_file), "missing the MTX file");
    ERR_RET(!file_exists(row_file), "missing the ROW file");
    ERR_RET(!file_exists(col_file), "missing the COL file");
    ERR_RET(!file_exists(select_file), "missing the SELECTED file");

    // First pass: select rows and create temporary MTX
    copy_selected_rows(mtx_file, row_file, select_file, output + "-temp");

    std::string temp_mtx_file = output + "-temp.mtx.gz";

    // Second pass: squeeze out empty columns
    filter_col_by_nnz(1, temp_mtx_file, col_file, output);

    if (file_exists(temp_mtx_file)) {
        std::remove(temp_mtx_file.c_str());
    }

    std::string out_row_file = output + ".rows.gz";

    if (file_exists(out_row_file)) {
        std::string temp_row_file = output + ".rows.gz-backup";
        WLOG("Remove existing output row file: " << out_row_file);
        copy_file(out_row_file, temp_row_file);
        remove_file(out_row_file);
        // std::filesystem::copy(out_row_file.c_str(), temp_row_file.c_str());
        // std::remove(out_row_file.c_str());
    }

    {
        std::string temp_row_file = output + "-temp.rows.gz";
        rename_file(temp_row_file, out_row_file);
        // std::filesystem::rename(temp_row_file.c_str(), out_row_file.c_str());
    }

    return EXIT_SUCCESS;
}
