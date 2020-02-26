#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>

#include "inference/representation_gaussian.hh"
#include "utils/util.hh"

#ifndef SGD_REGRESSION_HH_
#define SGD_REGRESSION_HH_

template <typename repr_t, typename param_t>
struct sgd_regression_t {
    using data_t = typename repr_t::data_t;
    using scalar_t = typename repr_t::scalar_t;
    using index_t = typename repr_t::index_t;

    explicit sgd_regression_t(param_t &theta)
        : p(theta.rows())
        , m(theta.cols())
        , eta(1, m)
    {
        //
    }

    // sample

    ////////////////////////
    // stochastic average //
    ////////////////////////

    // add_sgd(llik) // llik = 1 x m, update_repr

    // eval_sgd(), eta.summarize()
    // -> eta.get_grad_type1() -> G1 [1 x m]
    // -> eta.get_grad_type2() -> G2 [1 x m]
    // param_eval_sgd

    ///////////////////////
    // minibatch average //
    ///////////////////////

    // XtG1 += x * G1 [p x m]
    // X2tG2 += x2 * G2 [p x m]
    // nobs += 1.0

    /////////////////////////////
    // back-prop to parameters //
    /////////////////////////////

    // param_eval_sgd(XtG1, X2tG2, nobs)

    const index_t p;
    const index_t m;

    repr_t eta; // 1 x m lightweight representation
};

#endif
