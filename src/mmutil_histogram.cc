#include "mmutil_histogram.hh"

void
print_help(const char *fname)
{
    const char *_desc = "[Arguments]\n"
                        "MTX:    Input matrix market file\n"
                        "OUTPUT: Output file name (or STDOUT)\n"
                        "\n";

    std::cerr << "Take the histogram of MTX" << std::endl;
    std::cerr << std::endl;
    std::cerr << fname << " MTX OUTPUT" << std::endl;
    std::cerr << std::endl;
    std::cerr << _desc << std::endl;
}

int
main(const int argc, const char *argv[])
{
    if (argc < 3) {
        print_help(argv[0]);
        return EXIT_FAILURE;
    }

    const std::string mtx_file(argv[1]);
    const std::string output(argv[2]);

    CHECK(write_histogram_results(mtx_file, output));
}
