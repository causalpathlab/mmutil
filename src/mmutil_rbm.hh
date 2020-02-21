#include "mmutil.hh"
#include "inference/sampler.hh"
#include "inference/adam.hh"

#ifndef MMUTIL_RBM_HH_
#define MMUTIL_RBM_HH_

struct rbm_t {

  explicit rbm_t () {}

  Mat bias_vis;
  Mat bias_hid;       // 
  Mat weight_vis_hid; // #visible x #hidden

  Mat hidden;         // #batch x #hidden
  Mat visible;        // #

  
};


  // using DS = discrete_sampler_t<Scalar, Index>;
  // DS sampler_k(mu.cols()); // sample discrete from log-mass




  // Mat ; // label x cell





#endif
