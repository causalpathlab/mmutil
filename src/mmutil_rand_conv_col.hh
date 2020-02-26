#include <random>

#include "mmutil.hh"

#ifndef MMUTIL_CONV_COL_HH_
#define MMUTIL_CONV_COL_HH_

struct NrefT {
    explicit NrefT(const Index _val)
        : val(_val)
    {
    }
    const Index val;
};
struct ConvSampleT {
    explicit ConvSampleT(const Index _val)
        : val(_val)
    {
    }
    const Index val;
};
struct RefPerSampleT {
    explicit RefPerSampleT(const Index _val)
        : val(_val)
    {
    }
    const Index val;
};

SpMat
sample_conv_index(const NrefT _Nref, //
                  const ConvSampleT _N, //
                  const RefPerSampleT _D)
{
    std::random_device rd;
    std::mt19937 rng(rd());

    const Index Nref = _Nref.val;
    const Index N = _N.val;
    const Index D = _D.val;

    std::vector<Index> ref(Nref);
    std::iota(ref.begin(), ref.end(), 0);

    Index s = 0;
    using ETriplet = Eigen::Triplet<Scalar>;
    using ETripletVec = std::vector<ETriplet>;

    ETripletVec triplets;

    for (Index i = 0; i < N; ++i) {
        for (Index d = 0; d < D; ++d) {
            if ((s % Nref) == 0) {
                std::shuffle(ref.begin(), ref.end(), rng);
            }
            const Index j = ref.at(s % Nref);
            triplets.push_back(ETriplet(j, i, 1.0));
            s++;
        }
    }

    SpMat C(Nref, N);
    C.reserve(triplets.size());
    C.setFromTriplets(triplets.begin(), triplets.end());

    return C;
}

#endif
