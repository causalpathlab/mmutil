#include "mmutil_index.hh"
#include "utils/bgzstream.hh"
#include "mmutil_util.hh"

void
print_help(const std::string exe)
{
    const char *_usage = "\n"
                         "MTX   : Matrix market file (block-gzipped)\n"
                         "INDEX : Index file (default: ${MTX}.index)\n"
                         "\n";

    std::cerr << exe << " MTX [INDEX]" << _usage << std::endl;
}

int
main(const int argc, const char *argv[])
{
    if (argc < 2) {
        print_help(argv[0]);
        return EXIT_FAILURE;
    }

    std::string mtx_file = argv[1];

    if (!is_file_bgz(mtx_file)) {
        convert_bgzip(mtx_file);
    }

    std::string index_file = argv[2];
    if (argc > 2) {
        index_file = mtx_file + ".index";
    }

    CHECK(build_mmutil_index(mtx_file, index_file));

    TLOG("Done");
    return EXIT_SUCCESS;
}
