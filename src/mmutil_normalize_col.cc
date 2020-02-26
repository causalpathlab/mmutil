#include "mmutil_normalize.hh"

void
print_help(const char *fname)
{
    const char *_norm_help =
        "[Arguments]\n"
        "\n"
        "TAU: Regularization scaling (TAU = 1 works well)\n"
        "MTX: Input file (matrix market format)\n"
        "OUT: Output file (matrix market format)\n"
        "\n"
        "[Details]\n"
        "\n"
        "X(i,j) <- X(i,j) / sqrt{ d(j) + tau * mean(d(j)) }\n"
        "\n"
        "where d(j) = sum X(i,j)^2 / (sum X(i,j))^2\n"
        "\n"
        "This will results in a regularized shared-neighbor graph\n"
        "S(j,k) <- sum X(i,j) X(i,k) = D^(-1/2) A(j,k) D^(-1/2)\n"
        "\n";

    std::cerr << "Normalize the columns of sparse count matrix file."
              << std::endl;
    std::cerr << fname << " mmutil_normalize_col TAU MTX OUT\n" << std::endl;
    std::cerr << _norm_help << std::endl;
}

int
main(const int argc, const char *argv[])
{
    if (argc < 4) {
        print_help(argv[0]);
        return EXIT_FAILURE;
    }

    using Str = std::string;

    const Scalar tau_scale = std::stof(argv[1]);
    const Str mtx_file(argv[2]);
    const Str out_file(argv[3]);

    const Str _out_file = is_file_gz(out_file) ? out_file : (out_file + ".gz");
    write_normalized(mtx_file, _out_file, tau_scale);
    return EXIT_SUCCESS;
}
