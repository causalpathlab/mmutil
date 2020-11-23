#include <getopt.h>

#include "mmutil_annotate.hh"

#ifndef MMUTIL_ANNOTATE_COL_
#define MMUTIL_ANNOTATE_COL_

//////////////////////
// Argument parsing //
//////////////////////

template <typename T>
int
parse_annotation_options(const int argc,     //
                         const char *argv[], //
                         T &options)
{
    const char *_usage =
        "\n"
        "[Arguments]\n"
        "--mtx (-m)             : data MTX file\n"
        "--data (-m)            : data MTX file\n"
        "--col (-c)             : data column file\n"
        "--feature (-f)         : data row file (features)\n"
        "--row (-f)             : data row file (features)\n"
        "--ann (-a)             : row annotation file; each line contains a tuple of feature and label\n"
        "--anti (-A)            : row anti-annotation file; each line contains a tuple of feature and label\n"
        "--qc (-q)              : row annotation file for Q/C; each line contains a tuple of feature and minimum score\n"
        "--out (-o)             : Output file header\n"
        "\n"
        "--log_scale (-L)       : Data in a log-scale (default: false)\n"
        "--raw_scale (-R)       : Data in a raw-scale (default: true)\n"
        "\n"
        "--batch_size (-B)      : Batch size (default: 100000)\n"
        "--kappa_max (-K)       : maximum kappa value (default: 100)\n"
        "\n"
        "--em_iter (-i)         : EM iteration (default: 100)\n"
        "--em_tol (-t)          : EM convergence criterion (default: 1e-4)\n"
        "\n"
        "--verbose (-v)         : Set verbose (default: false)\n"
        "--output_mtx_file (-O) : Write a count matrix of the markers (default: false)\n"
        "\n";

    const char *const short_opts = "m:c:f:a:A:o:I:B:K:M:LRi:t:hbd:u:r:l:kOv";

    const option long_opts[] =
        { { "mtx", required_argument, nullptr, 'm' },            //
          { "data", required_argument, nullptr, 'm' },           //
          { "col", required_argument, nullptr, 'c' },            //
          { "row", required_argument, nullptr, 'f' },            //
          { "feature", required_argument, nullptr, 'f' },        //
          { "ann", required_argument, nullptr, 'a' },            //
          { "anti", required_argument, nullptr, 'A' },           //
          { "qc", required_argument, nullptr, 'q' },             //
          { "out", required_argument, nullptr, 'o' },            //
          { "log_scale", no_argument, nullptr, 'L' },            //
          { "raw_scale", no_argument, nullptr, 'R' },            //
          { "batch_size", required_argument, nullptr, 'B' },     //
          { "kappa_max", required_argument, nullptr, 'K' },      //
          { "em_iter", required_argument, nullptr, 'i' },        //
          { "em_tol", required_argument, nullptr, 't' },         //
          { "help", no_argument, nullptr, 'h' },                 //
          { "output_mtx_file", no_argument, nullptr, 'O' },      //
          { "verbose", no_argument, nullptr, 'v' },              //
          { nullptr, no_argument, nullptr, 0 } };

    while (true) {
        const auto opt = getopt_long(argc,                      //
                                     const_cast<char **>(argv), //
                                     short_opts,                //
                                     long_opts,                 //
                                     nullptr);

        if (-1 == opt)
            break;

        switch (opt) {
        case 'm':
            options.mtx = std::string(optarg);
            break;

        case 'c':
            options.col = std::string(optarg);
            break;

        case 'f':
            options.row = std::string(optarg);
            break;

        case 'a':
            options.ann = std::string(optarg);
            break;

        case 'A':
            options.anti_ann = std::string(optarg);
            break;

        case 'q':
            options.qc_ann = std::string(optarg);
            break;

        case 'o':
            options.out = std::string(optarg);
            break;

        case 'i':
            options.max_em_iter = std::stoi(optarg);
            break;

        case 't':
            options.em_tol = std::stof(optarg);
            break;

        case 'B':
            options.batch_size = std::stoi(optarg);
            break;
        case 'L':
            options.log_scale = true;
            options.raw_scale = false;
            break;
        case 'R':
            options.log_scale = false;
            options.raw_scale = true;
            break;
        case 'v': // -v or --verbose
            options.verbose = true;
            break;
        case 'K':
            options.kappa_max = std::stof(optarg);
            break;
        case 'O':
            options.output_count_matrix = true;
            break;
        case 'h': // -h or --help
        case '?': // Unrecognized option
            std::cerr << _usage << std::endl;
            return EXIT_FAILURE;
        default: //
                 ;
        }
    }

    ERR_RET(!file_exists(options.mtx), "No MTX data file");
    ERR_RET(!file_exists(options.col), "No COL data file");
    ERR_RET(!file_exists(options.row), "No ROW data file");
    ERR_RET(!file_exists(options.ann), "No ANN data file");

    return EXIT_SUCCESS;
}

#endif
