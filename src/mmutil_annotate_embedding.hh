#include <getopt.h>

#include "mmutil.hh"
#include "io.hh"
#include "mmutil_embedding.hh"

#ifndef MMUTIL_ANNOTATE_EMBEDDING_HH_
#define MMUTIL_ANNOTATE_EMBEDDING_HH_

struct embedding_options_t {
    using Str = std::string;

    typedef enum { UNIFORM, CV, MEAN } sampling_method_t;
    const std::vector<Str> METHOD_NAMES;

    embedding_options_t()
    {
        out = "output.txt.gz";
        embedding_dim = 2;
        embedding_epochs = 1000;
        exaggeration = 100;
        tol = 1e-8;
        verbose = false;
        l2_penalty = 1e-4;
    }

    Str data_file;
    Str prob_file;

    Str out;

    Index embedding_dim;
    Index embedding_epochs;
    Index exaggeration;

    Scalar tol;
    Scalar l2_penalty;

    bool verbose;
};

template <typename T>
int
parse_embedding_options(const int argc,     //
                        const char *argv[], //
                        T &options)
{
    const char *_usage =
        "\n"
        "[Arguments]\n"
        "--data (-m)             : data matrix file\n"
        "--prob (-p)             : probability matrix file\n"
        "--out (-o)              : Output file header\n"
        "\n"
        "--embedding_dim (-d)    : latent dimensionality (default: 2)\n"
        "--embedding_epochs (-i) : Maximum iteration (default: 100)\n"
        "--l2 (-l)               : L2 penalty (default: 1e-4)\n"
        "--tol (-t)              : Convergence criterion (default: 1e-4)\n"
        "--verbose (-v)          : Set verbose (default: false)\n"
        "\n";

    const char *const short_opts = "m:p:o:d:i:t:l:hv";

    const option long_opts[] =
        { { "data", required_argument, nullptr, 'm' },             //
          { "prob", required_argument, nullptr, 'p' },             //
          { "out", required_argument, nullptr, 'o' },              //
          { "embedding_dim", required_argument, nullptr, 'd' },    //
          { "embed_dim", required_argument, nullptr, 'd' },        //
          { "dim", required_argument, nullptr, 'd' },              //
          { "embedding_epochs", required_argument, nullptr, 'i' }, //
          { "l2", required_argument, nullptr, 'l' },               //
          { "l2_penalty", required_argument, nullptr, 'l' },       //
          { "tol", required_argument, nullptr, 't' },              //
          { "verbose", no_argument, nullptr, 'v' },                //
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
            options.data_file = std::string(optarg);
            break;

        case 'p':
            options.prob_file = std::string(optarg);
            break;

        case 'o':
            options.out = std::string(optarg);
            break;

        case 'd':
            options.embedding_dim = std::stoi(optarg);
            break;

        case 'i':
            options.embedding_epochs = std::stoi(optarg);
            break;

        case 't':
            options.tol = std::stof(optarg);
            break;

        case 'l':
            options.l2_penalty = std::stof(optarg);
            break;

        case 'v': // -v or --verbose
            options.verbose = true;
            break;

        case 'h': // -h or --help
        case '?': // Unrecognized option
            std::cerr << _usage << std::endl;
            return EXIT_FAILURE;
        default: //
                 ;
        }
    }

    ERR_RET(!file_exists(options.data_file), "No data matrix file");
    ERR_RET(!file_exists(options.prob_file), "No probability file");

    return EXIT_SUCCESS;
}

#endif
