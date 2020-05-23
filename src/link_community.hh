#include "mmutil.hh"

#ifndef LINK_COMMUNITY_HH_
#define LINK_COMMUNITY_HH_

struct nvertex_t : public check_positive_t<Index> {
    explicit nvertex_t(const Index n)
        : check_positive_t<Index>(n)
    {
    }
};

struct nedge_t : public check_positive_t<Index> {
    explicit nedge_t(const Index n)
        : check_positive_t<Index>(n)
    {
    }
};

struct ncolour_t : public check_positive_t<Index> {
    explicit ncolour_t(const Index n)
        : check_positive_t<Index>(n)
    {
    }
};

struct ngibbs_t : public check_positive_t<Index> {
    explicit ngibbs_t(const Index n)
        : check_positive_t<Index>(n)
    {
    }
};

struct nvb_t : public check_positive_t<Index> {
    explicit nvb_t(const Index n)
        : check_positive_t<Index>(n)
    {
    }
};

struct nburnin_t : public check_positive_t<Index> {
    explicit nburnin_t(const Index n)
        : check_positive_t<Index>(n)
    {
    }
};

struct nlocal_t : public check_positive_t<Index> {
    explicit nlocal_t(const Index n)
        : check_positive_t<Index>(n)
    {
    }
};

struct lc_model_t {

    explicit lc_model_t(const nvertex_t vert,
                        const nedge_t edge,
                        const ncolour_t col)
        : n(vert.val)
        , m(edge.val)
        , K(col.val)
        , Y(m, n)
        , Z(K, m)
        , Deg(K, n)
        , Prop(K, n)
        , lnProp(K, n)
        , Tot(K)
        , a0(1e-4)
        , b0(1e-4)
    {
#ifdef DEBUG
        TLOG("Y: " << Y.rows() << " x " << Y.cols());
        TLOG("Z: " << Z.rows() << " x " << Z.cols());
        TLOG("Deg: " << Deg.rows() << " x " << Deg.cols());
        TLOG("Prop: " << Prop.rows() << " x " << Prop.cols());
        TLOG("Tot: " << Tot.rows() << " x " << Tot.cols());
#endif
    }

    const Index n; // #vertex
    const Index m; // #edge
    const Index K; // #colour

    SpMat Y;    // edge x vertex incidence matrix
    Mat Z;      // edge x colour latent matrix
    Mat Deg;    // colour x vertex degree matrix
    Mat Prop;   // colour x vertex propensity matrix
    Mat lnProp; // colour x vertex propensity matrix
    Vec Tot;    // colour x 1

    const Scalar a0;
    const Scalar b0;
};

template <typename TVEC>
std::shared_ptr<lc_model_t>
build_lc_model(const TVEC &knn_index, const Index K)
{

    Index n = 0;
    Index m = 0;

    using Triplet = Eigen::Triplet<Scalar>;
    std::vector<Triplet> triplets;
    triplets.reserve(knn_index.size());

    auto _fun = [&](const auto &tt) {
        Index i, j;
        Scalar w;
        std::tie(i, j, w) = tt;

        if (i >= n)
            n = i + 1;
        if (j >= n)
            n = j + 1;
        triplets.emplace_back(Triplet(m, i, w));
        triplets.emplace_back(Triplet(m, j, w));
        ++m;
    };

    std::for_each(knn_index.begin(), knn_index.end(), _fun);

    std::shared_ptr<lc_model_t> ret =
        std::make_shared<lc_model_t>(nvertex_t(n), nedge_t(m), ncolour_t(K));

    lc_model_t &lc = *ret.get();

    // Populate elements in the data matrix
    lc.Y.reserve(triplets.size());

    TLOG(lc.Y.rows() << " x " << lc.Y.cols());

    lc.Y.setFromTriplets(triplets.begin(), triplets.end());

    // Zeroing sufficient statistics
    lc.Z.setZero();
    lc.Deg.setZero();
    lc.Tot.setZero();
    lc.Prop.setZero();

    return ret;
}

inline void
update_latent_random(lc_model_t &lc, const Index kmin, const Index kmax)
{
    const Index K = std::min(lc.K, kmax);
    const Index m = lc.m;
    lc.Z.setZero();
#ifdef DEBUG
    ASSERT(kmax > 0, "link_community: Kmax must be a positive integer");
    ASSERT(kmin >= 0, "link_community: Kmin must be non-negative");
#endif
    if (kmin >= K) {
        for (Index e = 0; e < m; ++e) {
            lc.Z(kmin, e) = 1.;
        }
        return;
    }

    using DS = discrete_sampler_t<Scalar, Index>;
    DS sampler(K - kmin); // sample discrete from log-mass
    Vec mass(K - kmin);
    mass.setZero();

    for (Index e = 0; e < m; ++e) {
        const Index k = kmin + sampler(mass);
        lc.Z(k, e) = 1.;
    }
}

inline void
update_param_fixed(lc_model_t &lc)
{
    const Index n = lc.n;
    lc.Deg = lc.Z * lc.Y;
    lc.Tot = lc.Deg * Mat::Ones(n, 1);

#ifdef DEBUG
    ASSERT(lc.Deg.minCoeff() > -1e-4, "link_community: update_param_fixed: found negative deg");
#endif

    // Ball, Karrer & Newman (2011)
    auto _prop = [](const Scalar &dd, const Scalar &tt) -> Scalar {
        return dd / (std::sqrt(tt) + 1e-8);
    };

    lc.Prop.setZero();
    for (Index j = 0; j < lc.Deg.cols(); ++j) {
        lc.Prop.col(j) += lc.Deg.col(j).binaryExpr(lc.Tot, _prop);
    }

    lc.Tot = lc.Prop * Mat::Ones(n, 1);
}

/** Update parameters by Variational Bayes
 * @param lc
 * @param clamp
 * @param nlocal
 */
template <typename Set>
inline Scalar
update_param_vb(lc_model_t &lc, const Index nlocal, const Set &clamp)
{
    const Index n = lc.n;
    const Index m = lc.m;
    const Index K = lc.K;

    lc.Deg = lc.Z * lc.Y;
    lc.Tot = lc.Prop * Mat::Ones(n, 1);

#ifdef DEBUG
    ASSERT(lc.Deg.minCoeff() > -1e-4, "link_community: update_param_vb: found negative deg");
#endif

    const Scalar a0 = lc.a0, b0 = lc.b0;

    auto _mf = [&](const Scalar &dd, const Scalar &tt) {
        return (dd + a0) / (b0 + tt);
    };

    auto _ln_mf = [&](const Scalar &dd, const Scalar &tt) {
        return fasterdigamma(dd + a0) - fasterlog(b0 + tt);
    };

    for (Index iter = 0; iter < nlocal; ++iter) { //
        for (Index j = 0; j < n; ++j) {           // O(nK)
            lc.Tot -= lc.Prop.col(j);
            lc.Prop.col(j) = lc.Deg.col(j).binaryExpr(lc.Tot, _mf);
            lc.lnProp.col(j) = lc.Deg.col(j).binaryExpr(lc.Tot, _ln_mf);
            lc.Tot += lc.Prop.col(j);
        }
    }

#ifdef DEBUG
    ASSERT(m == lc.Y.rows(), "link_community: Must have matching #rows");
    ASSERT(m == lc.Y.outerSize(), "link_community: Must be rowmajor SpMat");
#endif
    Scalar score = 0.;

    Vec mass(K);
    Vec z(K);

    for (Index e = 0; e < m; ++e) {
        if (clamp.count(e) > 0)
            continue;

        // lc.Deg -= lc.Z.col(e) * lc.Y.row(e); ~ O(1)
        for (SpMat::InnerIterator it(lc.Y, e); it; ++it) {
            const Index i = it.col();
            const Scalar y = it.value();
            lc.Deg.col(i) -= lc.Z.col(e) * y; // ~ O(K)
#ifdef DEBUG
	    ASSERT(lc.Deg.col(i).minCoeff() > -1e-4,
		   "delta-update gibbs, deg");
#endif
        }

        lc.Z.col(e).setZero();
        mass.setZero();
        for (SpMat::InnerIterator it(lc.Y, e); it; ++it) { // O(1)
            const Index i = it.col();
            mass += lc.lnProp.col(i); // O(K)
        }

        normalized_exp(mass, z);
        lc.Z.col(e) = z;
        score += mass.cwiseProduct(z).sum();

        // lc.Deg += lc.Z.col(e) * lc.Y.row(e); ~ O(1)
        for (SpMat::InnerIterator it(lc.Y, e); it; ++it) {
            const Index i = it.col();
            const Scalar y = it.value();
            lc.Deg.col(i) += lc.Z.col(e) * y; // ~ O(K)
        }
    }

    return score;
}

/** Update parameters by Gibbs sampling
 * @param lc
 * @param clamp
 * @param nlocal
 */
template <typename Set>
inline Scalar
update_param_gibbs(lc_model_t &lc, const Index nlocal, const Set &clamp)
{
    const Index n = lc.n;
    const Index m = lc.m;
    const Index K = lc.K;

    lc.Deg = lc.Z * lc.Y;
    lc.Tot = lc.Prop * Mat::Ones(n, 1);
#ifdef DEBUG
    ASSERT(lc.Deg.minCoeff() > -1e-4, "link_community: update_param_gibbs: found negative deg");
#endif

    std::random_device rd;
    std::mt19937 gen(rd());
    using gam_t = std::gamma_distribution<Scalar>;
    using param_t = gam_t::param_type;

    const Scalar a0 = lc.a0, b0 = lc.b0;

    auto _rgam = [&](const Scalar &dd, const Scalar &tt) {
        gam_t g(dd + a0, 1.0 / (b0 + tt));
        return g(gen);
    };

    for (Index iter = 0; iter < nlocal; ++iter) { //
        for (Index j = 0; j < n; ++j) {           // O(nK)
            lc.Tot -= lc.Prop.col(j);
            lc.Prop.col(j) = lc.Deg.col(j).binaryExpr(lc.Tot, _rgam);
            lc.Tot += lc.Prop.col(j);
        }
    }

#ifdef DEBUG
    ASSERT(m == lc.Y.rows(), "link_community: Must have matching #rows");
    ASSERT(m == lc.Y.outerSize(), "link_community: Must be rowmajor SpMat");
#endif

    using DS = discrete_sampler_t<Scalar, Index>;
    DS sampler(K); // sample discrete from log-mass
    Vec mass(K);

    auto log_op = [](const Scalar &x) -> Scalar {
        const Scalar pmin = 1e-16;
        return fasterlog(std::max(x, pmin));
    };

    lc.lnProp = lc.Prop.unaryExpr(log_op);

    Scalar score = 0.;

    for (Index e = 0; e < m; ++e) {
        if (clamp.count(e) > 0)
            continue;

        // lc.Deg -= lc.Z.col(e) * lc.Y.row(e); ~ O(1)
        for (SpMat::InnerIterator it(lc.Y, e); it; ++it) {
            const Index i = it.col();
            const Scalar y = it.value();
            lc.Deg.col(i) -= lc.Z.col(e) * y; // ~ O(K)
#ifdef DEBUG
	    ASSERT(lc.Deg.col(i).minCoeff() > -1e-4,
		   "delta-update gibbs, deg");
#endif

        }

        lc.Z.col(e).setZero();
        mass.setZero();
        for (SpMat::InnerIterator it(lc.Y, e); it; ++it) { // O(1)
            const Index i = it.col();
            mass += lc.lnProp.col(i); // O(K)
        }
        const Index k = sampler(mass); // ~ O(K)
        score += mass(k);
        lc.Z(k, e) = 1.;

        // lc.Deg += lc.Z.col(e) * lc.Y.row(e); ~ O(1)
        for (SpMat::InnerIterator it(lc.Y, e); it; ++it) {
            const Index i = it.col();
            const Scalar y = it.value();
            lc.Deg.col(i) += lc.Z.col(e) * y; // ~ O(K)
        }
    }

    return score;
}

/**
   @param lc
   @param clamp
   @param ngibbs
   @param nburnin
   @param nlocal
 */
template <typename Set, typename FUN>
void
run_gibbs_sampling(lc_model_t &lc,
                   const Set &clamp,
                   FUN &stat_fun,
                   const ngibbs_t ngibbs,
                   const nburnin_t nburnin,
                   const nlocal_t nlocal)
{
    TLOG("Initialize parameters");
    update_param_fixed(lc);

    // std::cout << std::endl << lc.Deg.transpose() << std::endl;
    // std::cout << std::endl;

    TLOG("Start Gibbs sampling...");
    for (Index iter = 0; iter < ngibbs.val + nburnin.val; ++iter) {
        const Scalar score = update_param_gibbs(lc, nlocal.val, clamp);

        std::cerr << "\r"
                  << "[" << std::setw(10) << iter << "][" << std::setw(10)
                  << (score / static_cast<Scalar>(lc.n)) << "]";

        if (iter >= nburnin.val) {
            stat_fun(lc);
        }
    }
    std::cerr << std::endl;
    TLOG("Done Gibbs sampling");
}

/**
   @param lc
   @param clamp
   @param ngibbs
   @param nburnin
   @param nlocal
 */
template <typename FUN>
void
run_gibbs_sampling(lc_model_t &lc,
                   FUN &stat_fun,
                   const ngibbs_t ngibbs,
                   const nburnin_t nburnin,
                   const nlocal_t nlocal)
{
    std::unordered_set<Index> empty;
    empty.clear();
    run_gibbs_sampling(lc, empty, stat_fun, ngibbs, nburnin, nlocal);
}

/**
   @param lc
   @param clamp
   @param nvb
   @param nlocal
 */
template <typename Set>
void
run_vb_optimization(lc_model_t &lc,
                    const Set &clamp,
                    const nvb_t nvb,
                    const nlocal_t nlocal)
{
    Scalar diff, score_prev = 0.;
    bool conv = false;
    TLOG("Start VB optimization...");
    for (Index iter = 0; iter < nvb.val; ++iter) {

        const Scalar score = update_param_vb(lc, nlocal.val, clamp);

        diff = std::abs(score - score_prev) / std::abs(score_prev + 1e-4);

        if (diff < 1e-4) {
            conv = true;
            break;
        }

        score_prev = score;
        std::cerr << "\r"
                  << "[" << std::setw(10) << iter << "][" << std::setw(10)
                  << (score / static_cast<Scalar>(lc.n)) << "]";
    }
    std::cerr << std::endl;
    TLOG("Done VB optimization [convergence: " << std::setw(10) << diff << "]");

    const Scalar a0 = lc.a0, b0 = lc.b0;

    auto _mean = [&](const Scalar &dd, const Scalar &tt) {
        return (dd + a0) / (b0 + tt);
    };

    auto _sd = [&](const Scalar &dd, const Scalar &tt) {
        return std::sqrt(dd + a0) / (b0 + tt);
    };
}

/**
   @param lc
   @param nvb
   @param nlocal
 */
void
run_vb_optimization(lc_model_t &lc, const nvb_t nvb, const nlocal_t nlocal)
{
    std::unordered_set<Index> empty;
    empty.clear();
    run_vb_optimization(lc, empty, nvb, nlocal);
}
#endif
