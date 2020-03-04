#include "mmutil_annotate_embedding.hh"

int
main(const int argc, const char *argv[])
{

    embedding_options_t options;
    CHECK(parse_embedding_options(argc, argv, options));

    Mat X;  // N x D
    Mat Pr; // N x K

    CHECK(read_data_file(options.prob_file, Pr));
    CHECK(read_data_file(options.data_file, X));

    Mat Y;
    Mat phi;
    std::tie(Y, phi) = run_cluster_embedding(Pr, X, options);

    write_data_file(options.out + ".embedding.gz", Y);
    write_data_file(options.out + ".phi.gz", phi);

    return EXIT_SUCCESS;
}
