#include <algorithm>
#include <functional>

#include "component_gaussian.hh"
#include "dpm.hh"
#include "mmutil.hh"
#include "sampler.hh"

#ifndef MMUTIL_CLUSTER_HH_
#define MMUTIL_CLUSTER_HH_

// void fit_gadpm

void fit(const Mat& X) {

  const Index K = 10;

  using F0 = trunc_dpm_t<Vec>;
  using F  = multi_gaussian_component_t<Mat>;
  F::dim_t dim(X.rows());

  F0::dpm_alpha_t dpm_alpha(1.0);
  F0::num_clust_t num_clust(K);

  F0 dpm(dpm_alpha, num_clust);

  std::vector<Index> cindex(K);
  std::iota(cindex.begin(), cindex.end(), 0);
  std::vector<F> components;
  std::transform(cindex.begin(), cindex.end(), std::back_inserter(components),
                 [&dim](const auto&) { return F(dim); });

  std::for_each(components.begin(), components.end(),
                [&X](F& f) { std::cout << f.log_lcvi(X.col(0)) << std::endl; });


  // dpm.log_lcvi();
  
  // SpMat Z;


  // 1. Random seeding

  // 2. Initialization by collapsed Gibbs 


  // 3. Minibatch optimization

  // TODO: fixed membership

}

#endif
