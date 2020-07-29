#include <algorithm>
#include <random>
#include <unordered_map>
#include <unordered_set>

#include "eigen_util.hh"
#include "inference/sampler.hh"
#include "io.hh"
#include "mmutil.hh"
#include "mmutil_normalize.hh"
#include "mmutil_index.hh"
#include "utils/progress.hh"
#include "mmutil_bgzf_util.hh"
#include "mmutil_util.hh"
#include "ext/tabix/bgzf.h"

#ifndef MMUTIL_ANNOTATE_HH_
#define MMUTIL_ANNOTATE_HH_

struct annotation_model_t {

    explicit annotation_model_t(const Mat lab,
                                const Mat anti_lab,
                                const Scalar _kmax)
        : nmarker(lab.rows())
        , ntype(lab.cols())
        , mu(nmarker, ntype)
        , mu_anti(nmarker, ntype)
        , log_normalizer(ntype)
        , score(ntype)
        , kappa_init(1.0)  //
        , kappa_max(_kmax) //
        , kappa(kappa_init)
        , kappa_anti(kappa_init)
    {
        // initialization
        kappa = kappa_init;
        kappa_anti = kappa_init;
        log_normalizer.setZero();
        mu.setZero();
        mu_anti.setZero();
        mu += lab;
        mu_anti += anti_lab;
    }

    template <typename Derived, typename Derived2, typename Derived3>
    void update_param(const Eigen::MatrixBase<Derived> &_xsum,
                      const Eigen::MatrixBase<Derived2> &_xsum_anti,
                      const Eigen::MatrixBase<Derived3> &_nsum)
    {

        const Derived &Stat = _xsum.derived();
        const Derived2 &Stat_anti = _xsum_anti.derived();
        const Derived3 &nsize = _nsum.derived();

        //////////////////////////////////////////////////
        // concentration parameter for von Mises-Fisher //
        //////////////////////////////////////////////////

        // We use the approximation proposed by Banerjee et al. (2005) for
        // simplicity and relatively stable performance
        //
        //          (rbar*d - rbar^3)
        // kappa = -------------------
        //          1 - rbar^2

        const Scalar d = static_cast<Scalar>(nmarker);

        // We may need to share this kappa estimate across all the
        // types since some of the types might be
        // under-represented.

        const Scalar r = Stat.rowwise().sum().norm() / nsize.sum();

        Scalar _kappa = r * (d - r * r) / (1.0 - r * r);

        if (_kappa > kappa_max) {
            _kappa = kappa_max;
        }

        kappa = _kappa;

        const Scalar r0 = Stat_anti.rowwise().sum().norm() / nsize.sum();

        Scalar _kappa_anti = r0 * (d - r0 * r0) / (1.0 - r0 * r0);

        if (_kappa_anti > kappa_max) {
            _kappa_anti = kappa_max;
        }

        kappa_anti = _kappa_anti;

        ////////////////////////
        // update mean vector //
        ////////////////////////

        mu = Stat * nsize.cwiseInverse().asDiagonal();
        normalize_columns(mu);

        mu_anti = -Stat_anti * nsize.cwiseInverse().asDiagonal();
        normalize_columns(mu_anti);
        // update_log_normalizer();
    }

    template <typename Derived>
    inline const Vec &log_score(const Eigen::MatrixBase<Derived> &_x)
    {
        const Derived &x = _x.derived();
        score = (mu.transpose() * x) * kappa;
        score -= (mu_anti.transpose() * x) * kappa_anti;
        // score += log_normalizer; // not needed
        return score;
    }

    void update_log_normalizer()
    {
        // Normalizer for vMF
        //
        //            kappa^{d/2 -1}
        // C(kappa) = ----------------
        //            (2pi)^{d/2} I(d/2-1, kappa)
        //
        // where
        // I(v,x) = boost::math::cyl_bessel_i(v,x)
        //
        // ln C ~ (d/2 - 1) ln(kappa) - ln I(v, k)
        //

        const Scalar eps = 1e-8;
        const Scalar d = static_cast<Scalar>(nmarker);
        const Scalar df = d * 0.5 - 1.0 + eps;
        const Scalar ln2pi = std::log(2.0 * 3.14159265359);

        auto _log_denom = [&](const Scalar &kap) -> Scalar {
            Scalar ret = (0.5 * d - 1.0) * std::log(kap);
            ret -= ln2pi * (0.5 * d);
            ret -= _log_bessel_i(df, kap);
            return ret;
        };

        log_normalizer.setConstant(_log_denom(kappa));

        // std::cout << "\n\nnormalizer:\n"
        //           << log_normalizer.transpose() << std::endl;
    }

    const Index nmarker;
    const Index ntype;

    Mat mu;             // refined marker x type matrix
    Mat mu_anti;        // permuted marker x type matrix
    Vec log_normalizer; // log-normalizer
    Vec score;          // temporary score

    const Scalar kappa_init;
    const Scalar kappa_max;
    Scalar kappa;
    Scalar kappa_anti;
};

struct annotation_options_t {
    using Str = std::string;

    annotation_options_t()
    {
        mtx = "";
        col = "";
        row = "";
        ann = "";
        anti_ann = "";
        qc_ann = "";
        out = "output.txt.gz";

        col_norm = 10000;

        raw_scale = true;
        log_scale = false;

        batch_size = 100000;

        max_em_iter = 100;

        em_tol = 1e-4;
        kappa_max = 100.;

        balance_marker_size = false;
        unconstrained_update = false;
        output_count_matrix = false;

        verbose = false;
    }

    Str mtx;
    Str col;
    Str row;
    Str ann;
    Str anti_ann;
    Str qc_ann;
    Str out;

    Scalar col_norm;

    bool raw_scale;
    bool log_scale;

    Index batch_size;

    Index max_em_iter;

    Scalar em_tol;

    bool balance_marker_size;
    bool unconstrained_update;
    bool output_count_matrix;

    bool verbose;
    Scalar kappa_max;
};

template <typename T>
std::tuple<SpMat,
           SpMat,
           SpMat,
           std::vector<std::string>,
           std::vector<std::string>>
read_annotation_matched(const T &options)
{
    using Str = std::string;

    std::vector<std::tuple<Str, Str>> ann_pair_vec;
    if (options.ann.size() > 0) {
        read_pair_file<Str, Str>(options.ann, ann_pair_vec);
    }

    std::vector<std::tuple<Str, Str>> anti_pair_vec;
    if (options.anti_ann.size() > 0) {
        read_pair_file<Str, Str>(options.anti_ann, anti_pair_vec);
    }

    std::vector<std::tuple<Str, Scalar>> qc_pair_vec;
    if (options.qc_ann.size() > 0) {
        read_pair_file<Str, Scalar>(options.qc_ann, qc_pair_vec);
    }

    std::vector<Str> row_vec;
    read_vector_file(options.row, row_vec);

    std::unordered_map<Str, Index> row_pos; // row name -> row index
    for (Index j = 0; j < row_vec.size(); ++j) {
        if (row_pos.count(row_vec.at(j)) > 0) {
            WLOG("Duplicate row/feature name: " << row_vec.at(j));
            WLOG("Will ignore the previous one");
        }
        row_pos[row_vec.at(j)] = j;
    }

    std::unordered_map<Str, Index> label_pos; // label name -> label index
    {
        Index j = 0;
        for (auto pp : ann_pair_vec) {
            if (row_pos.count(std::get<0>(pp)) == 0)
                continue;
            if (label_pos.count(std::get<1>(pp)) == 0)
                label_pos[std::get<1>(pp)] = j++;
        }
        for (auto pp : anti_pair_vec) {
            if (row_pos.count(std::get<0>(pp)) == 0)
                continue;
            if (label_pos.count(std::get<1>(pp)) == 0)
                label_pos[std::get<1>(pp)] = j++;
        }
    }

    ASSERT(label_pos.size() > 0, "Insufficient #labels");

    using ET = Eigen::Triplet<Scalar>;
    std::vector<ET> triples;

    for (auto pp : ann_pair_vec) {
        if (row_pos.count(std::get<0>(pp)) > 0 &&
            label_pos.count(std::get<1>(pp)) > 0) {
            Index r = row_pos.at(std::get<0>(pp));
            Index l = label_pos.at(std::get<1>(pp));
            triples.push_back(ET(r, l, 1.0));
        }
    }

    std::vector<ET> anti_triples;
    for (auto pp : anti_pair_vec) {
        if (row_pos.count(std::get<0>(pp)) > 0 &&
            label_pos.count(std::get<1>(pp)) > 0) {
            Index r = row_pos.at(std::get<0>(pp));
            Index l = label_pos.at(std::get<1>(pp));
            anti_triples.push_back(ET(r, l, 1.0));
        }
    }

    std::vector<ET> qc_triples;
    for (auto pp : qc_pair_vec) {
        if (row_pos.count(std::get<0>(pp)) > 0) {
            Index r = row_pos.at(std::get<0>(pp));
            Scalar threshold = std::get<1>(pp);
            qc_triples.push_back(ET(r, 0, threshold));
        }
    }

    const Index max_rows = std::max(row_vec.size(), row_pos.size());
    const Index max_labels = label_pos.size();

    SpMat L(max_rows, max_labels);
    L.reserve(triples.size());
    L.setFromTriplets(triples.begin(), triples.end());

    SpMat L0(max_rows, max_labels);
    L0.reserve(anti_triples.size());
    L0.setFromTriplets(anti_triples.begin(), anti_triples.end());

    SpMat Lqc(max_rows, 1);
    Lqc.reserve(qc_triples.size());
    Lqc.setFromTriplets(qc_triples.begin(), qc_triples.end());

    std::vector<Str> labels(max_labels);
    std::vector<Str> rows(max_rows);

    for (auto pp : label_pos)
        labels[std::get<1>(pp)] = std::get<0>(pp);

    for (auto pp : row_pos)
        rows[std::get<1>(pp)] = std::get<0>(pp);

    if (options.verbose) {
        for (auto l : labels) {
            TLOG("Annotation Labels: " << l);
        }

        for (auto q : qc_pair_vec) {
            Index r = row_pos.at(std::get<0>(q));
            TLOG("Q/C: " << std::get<0>(q) << " " << r << " "
                         << std::get<1>(q));
        }
    }

    return std::make_tuple(L, L0, Lqc, rows, labels);
}

int
run_annotation(const annotation_options_t &options)
{
    //////////////////////////////////////////////////////////
    // Read the annotation information to construct initial //
    // type-specific marker gene profiles                   //
    //////////////////////////////////////////////////////////

    SpMat L_fg, L_bg; // gene x label
    SpMat L_qc;       // gene x 1

    std::vector<std::string> rows;
    std::vector<std::string> labels;

    std::tie(L_fg, L_bg, L_qc, rows, labels) = read_annotation_matched(options);

    std::vector<std::string> columns;
    CHK_ERR_RET(read_vector_file(options.col, columns),
                "Failed to read the column file: " << options.col);

    CHK_ERR_RET(mmutil::bgzf::convert_bgzip(options.mtx),
                "Failed to obtain a bgzipped file: " << options.mtx);

    std::string idx_file = options.mtx + ".index";
    CHK_ERR_RET(mmutil::index::build_mmutil_index(options.mtx, idx_file),
                "Failed to construct an index file: " << idx_file);

    std::string mtx_file = options.mtx;

    mm_info_reader_t info;
    CHECK(mmutil::bgzf::peek_bgzf_header(mtx_file, info));
    const Index D = info.max_row;
    const Index N = info.max_col;

    Index batch_size = options.batch_size;

    std::vector<Index> subrow;

    Vec nnz = L_fg * Mat::Ones(L_fg.cols(), 1) +
        L_bg.cwiseAbs() * Mat::Ones(L_bg.cols(), 1);

    for (SpMat::InnerIterator it(L_qc, 0); it; ++it) {
        const Index g = it.col();
        nnz(g) += 1;
    }

    for (Index r = 0; r < nnz.size(); ++r) {
        if (nnz(r) > 0)
            subrow.emplace_back(r);
    }

    auto log2_op = [](const Scalar &x) -> Scalar { return std::log2(1.0 + x); };

    std::vector<Index> idx_tab;
    CHECK(mmutil::index::read_mmutil_index(idx_file, idx_tab));

    auto take_batch_data_subcol = [&](std::vector<Index> &subcol,
                                      bool do_norm = true) -> Mat {
        SpMat x = mmutil::index::read_eigen_sparse_subset_row_col(mtx_file,
                                                                  idx_tab,
                                                                  subrow,
                                                                  subcol);

        if (options.log_scale) {
            x = x.unaryExpr(log2_op);
        }

        Mat xx = Mat(x);
        if (do_norm)
            normalize_columns(xx);
        return xx;
    };

    auto take_batch_data = [&](Index lb, Index ub, bool do_norm = true) -> Mat {
        std::vector<Index> subcol(ub - lb);
        std::iota(subcol.begin(), subcol.end(), lb);
        return take_batch_data_subcol(subcol, do_norm);
    };

    Mat L = row_sub(L_fg, subrow);
    Mat L0 = row_sub(L_bg, subrow);

    SpMat Lqc = row_sub(L_qc, subrow).transpose();

    annotation_model_t annot(L, L0, options.kappa_max);

    using DS = discrete_sampler_t<Scalar, Index>;

    ///////////////////////////////
    // Initial greedy assignment //
    ///////////////////////////////

    const Index M = L.rows();
    const Index K = L.cols();
    std::vector<Index> membership(N);
    std::fill(membership.begin(), membership.end(), -1);

    Vec xj(M);
    Vec nsize(K);
    Vec mass(K);

    const Scalar pseudo = 1e-8;
    Mat Stat(M, K);      // feature x label
    Mat Stat_anti(M, K); // feature x label
    nsize.setConstant(pseudo);
    Stat.setConstant(pseudo);      // sum x(g,j) * z(j, k)
    Stat_anti.setConstant(pseudo); // sum x0(g,j) * z(j, k)

    const Index max_em_iter = options.max_em_iter;
    Vec score_trace(max_em_iter);

    Mat mu = L;
    normalize_columns(mu);

    std::unordered_set<Index> taboo;

    // initial Q/C
    if (Lqc.cwiseAbs().sum() > 0) {

        for (Index lb = 0; lb < N; lb += batch_size) {
            const Index ub = std::min(N, batch_size + lb);

            Mat xx = take_batch_data(lb, ub);

            for (Index j = 0; j < xx.cols(); ++j) {
                const Index i = j + lb;
                xj = xx.col(j);
                for (SpMat::InnerIterator it(Lqc, 0); it; ++it) {

                    const Index g = it.col();
                    const Scalar v = it.value();

                    if (xj(g) < it.value()) {
                        taboo.insert(i);
                    }
                }
            }
        }
        TLOG("Found " << taboo.size() << " disqualified");
    }

    Scalar score_init = 0;

    auto find_argmax_membership = [&]() {
        for (Index lb = 0; lb < N; lb += batch_size) {
            const Index ub = std::min(N, batch_size + lb);
            Mat xx = take_batch_data(lb, ub);
            for (Index j = 0; j < xx.cols(); ++j) {
                const Index i = j + lb;
                if (taboo.count(i) > 0)
                    continue;
                xj = xx.col(j);
                Index argmax = 0;
                const Vec &_score = annot.log_score(xj);
                _score.maxCoeff(&argmax);
                membership[i] = argmax;
            }
        }
    };

    auto greedy_initialization = [&]() {
        // Traverse for each block
        for (Index lb = 0; lb < N; lb += batch_size) {
            const Index ub = std::min(N, batch_size + lb);

            Mat xx = take_batch_data(lb, ub);

            for (Index j = 0; j < xx.cols(); ++j) {
                const Index i = j + lb;
                if (taboo.count(i) > 0)
                    continue;
                xj = xx.col(j);
                Index argmax = 0;
                const Vec &_score = annot.log_score(xj);
                _score.maxCoeff(&argmax);
                score_init += _score(argmax);

                if (xj.sum() > 0) {
                    nsize(argmax) += 1.0;
                    Stat.col(argmax) += xj.cwiseProduct(L.col(argmax));
                    Stat_anti.col(argmax) += xj.cwiseProduct(L0.col(argmax));
                    membership[i] = argmax;
                } else {
                    taboo.insert(i);
                }
            }

            if (options.verbose) {
                std::cerr << nsize.transpose() << "\r" << std::flush;
            }
        }

        score_init /= static_cast<Scalar>(N);
        annot.update_param(Stat, Stat_anti, nsize);
    };

    ////////////////////////////
    // Memoized online update //
    ////////////////////////////

    auto score_diff = [&options](const Index iter, const Vec &trace) -> Scalar {
        Scalar diff = std::abs(trace(iter));

        if (iter > 4) {
            Scalar score_old = trace.segment(iter - 3, 2).sum();
            Scalar score_new = trace.segment(iter - 1, 2).sum();
            diff = std::abs(score_old - score_new) /
                (std::abs(score_old) + options.em_tol);
        } else if (iter > 0) {
            diff = std::abs(trace(iter - 1) - trace(iter)) /
                (std::abs(trace(iter) + options.em_tol));
        }

        return diff;
    };

    std::vector<Scalar> em_score_out;

    auto monte_carlo_update = [&]() {
        DS sampler_k(K); // sample discrete from log-mass
        Scalar score = score_init;
        Vec sj(K); // type x 1 score vector

        for (Index iter = 0; iter < max_em_iter; ++iter) {
            score = 0;

            for (Index lb = 0; lb < N; lb += batch_size) {     // batch
                const Index ub = std::min(N, batch_size + lb); //

                Mat xx = take_batch_data(lb, ub, true);

                for (Index j = 0; j < xx.cols(); ++j) {

                    const Index i = j + lb;

                    if (taboo.count(i) > 0)
                        continue;

                    xj = xx.col(j);

                    const Index k_prev = membership[i];
                    sj = annot.log_score(xj);
                    const Index k_now = sampler_k(sj);
                    score += sj(k_now);

                    if (k_now != k_prev) {

                        nsize(k_prev) -= 1.0;
                        nsize(k_now) += 1.0;

                        Stat.col(k_prev) -= xj.cwiseProduct(L.col(k_prev));
                        Stat.col(k_now) += xj.cwiseProduct(L.col(k_now));

                        Stat_anti.col(k_prev) -=
                            xj.cwiseProduct(L0.col(k_prev));
                        Stat_anti.col(k_now) += xj.cwiseProduct(L0.col(k_now));

                        membership[i] = k_now;
                    }

                    if (options.verbose) {
                        std::cerr << nsize.transpose() << "\r" << std::flush;
                    }
                } // end of data iteration
                annot.update_param(Stat, Stat_anti, nsize);
            } // end of batch iteration

            score = score / static_cast<Scalar>(N);
            score_trace(iter) = score;

            Scalar diff = score_diff(iter, score_trace);
            TLOG("Iter [" << iter << "] score = " << score
                          << ", diff = " << diff);

            if (iter > 4 && diff < options.em_tol) {

                TLOG("Converged < " << options.em_tol);

                for (Index t = 0; t <= iter; ++t) {
                    em_score_out.emplace_back(score_trace(t));
                }
                break;
            }
        } // end of EM iteration
    };

    TLOG("Start greedy initialization");
    greedy_initialization();
    TLOG("Finished greedy initialization");

    TLOG("Start training marker gene profiles");
    monte_carlo_update();
    TLOG("Finished training the main assignment model");

    TLOG("Writing the results ...");
    std::vector<std::string> markers;
    markers.reserve(subrow.size());
    std::for_each(subrow.begin(), subrow.end(), [&](const auto r) {
        markers.emplace_back(rows.at(r));
    });

    write_vector_file(options.out + ".marker_names.gz", markers);
    write_vector_file(options.out + ".label_names.gz", labels);
    write_data_file(options.out + ".marker_profile.gz", annot.mu);
    write_data_file(options.out + ".marker_profile_anti.gz", annot.mu_anti);
    write_vector_file(options.out + ".em_scores.gz", em_score_out);

    //////////////////////////////////////////////
    // Assign labels to all the cells (columns) //
    //////////////////////////////////////////////

    find_argmax_membership();

    using out_tup = std::tuple<std::string, std::string, Scalar, Scalar>;
    std::vector<out_tup> output;

    output.reserve(N);
    Vec zi(annot.ntype);
    Vec dbl_zi(annot.ntype);
    Mat Pr(annot.ntype, N);

    Pr.setZero();

    for (Index lb = 0; lb < N; lb += batch_size) {
        const Index ub = std::min(N, batch_size + lb);

        Mat xx = take_batch_data(lb, ub);

        for (Index j = 0; j < xx.cols(); ++j) {
            const Index i = j + lb;
            if (taboo.count(i) > 0) {
                output.emplace_back(columns.at(i), "Incomplete", 0., 0.);
                continue;
            }

            const Vec &_score = annot.log_score(xx.col(j));
            normalized_exp(_score, zi);

            Index argmax;
            const Scalar smax = _score.maxCoeff(&argmax);
            Pr.col(i) = zi;

            output.emplace_back(columns[i], labels[argmax], zi(argmax), smax);
        }
        TLOG("Annotated on the batch [" << lb << ", " << ub << ")");
    }

    Pr.transposeInPlace();
    write_tuple_file(options.out + ".annot.gz", output);
    write_data_file(options.out + ".annot_prob.gz", Pr);

    TLOG("Done");
    return EXIT_SUCCESS;
}

#endif
