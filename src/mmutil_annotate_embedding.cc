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

    // Q/C Pr matrix
    for (Index i = 0; i < Pr.rows(); ++i) {
        const Scalar pr_sum = Pr.row(i).sum();
        if (pr_sum > 1.0) {
            Pr.row(i) /= pr_sum;
        }
    }

    Mat xx = standardize(X);

    Mat Y;
    Mat phi;
    std::tie(Y, phi) = train_embedding_by_cluster(xx, Pr, options);
    Mat Ype = Pr * phi;

    write_data_file(options.out + ".pe.gz", Ype);
    write_data_file(options.out + ".embedding.gz", Y);
    write_data_file(options.out + ".phi.gz", phi);

    TLOG("Done");

    return EXIT_SUCCESS;
}
