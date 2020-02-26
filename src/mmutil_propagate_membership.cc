#include "mmutil_propagate_membership.hh"

void
print_help(const char *fname)
{
    const char *_desc =
        "Propagate membership of the second column to the first one.\n"
        "\n"
        "pi[i,k] = sum W[i,j] pi[j,k]\n"
        "\n"
        "[Arguments]\n"
        "MATCH      : Matching Results <i> <j> dist[i,j]\n"
        "MEMBERSHIP : Membership for the second column <j> <k>\n"
        "DECAY      : Decay parameter, W[i,j] = exp(-decay * dist[i,j]) \n"
        "OUTPUT     : Membership for the first column <i> <k>\n"
        "\n";
    std::cerr << _desc << std::endl;
    std::cerr << fname << " MATCH MEMBERSHIP OUTPUT" << std::endl;
    std::cerr << std::endl;
}

int
main(const int argc, const char *argv[])
{
    if (argc != 5) {
        print_help(argv[0]);
        return EXIT_FAILURE;
    }

    using Str = std::string;

    const Str match_file(argv[1]);
    const Str membership_file(argv[2]);
    const Scalar decay = std::stof(argv[3]);
    const Str out_file(argv[4]);

    auto out_vec = propagate_membership(match_file, membership_file, decay);

    write_tuple_file(out_file, out_vec);

    TLOG("Wrote a new membership file: " << out_file);

    return EXIT_SUCCESS;
}
