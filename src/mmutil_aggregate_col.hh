#include "mmutil.hh"
#include "mmutil_stat.hh"

#ifndef MMUTIL_AGGREGATE_COL_HH_
#define MMUTIL_AGGREGATE_COL_HH_

struct mat_stat_collector_t {

  using index_t  = Index;
  using scalar_t = Scalar;

  // Z : clusters x samples
  // X : dimension x samples

  explicit mat_stat_collector_t(const Mat& zz)
      : Z(zz),
        nn(Z.cols()),
        kk(Z.rows()) {}

  void set_dimension(const index_t D, const index_t n, const index_t e) {
    if (n > nn) WLOG("some samples will be ignored");
    S1.resize(kk, D);
    S2.resize(kk, D);
    N.resize(kk, D);
    z_j.resize(kk, 1);

    S1.setZero();
    S2.setZero();
    N.setZero();

    TLOG("Start aggregating the statistics: " << kk << " x " << D);
  }

  void eval(const index_t i, const index_t j, const scalar_t x_ij) {
    if (j < nn && j >= 0) {
      S1.col(i) += Z.col(j) * x_ij;
      S2.col(i) += Z.col(j) * x_ij * x_ij;
      N.col(i) += Z.col(j);
    }
  }

  void eval_end() { TLOG("Finished aggregating the statistics"); }

  const Mat& Z;
  const index_t nn;
  const index_t kk;

  Mat S1;
  Mat S2;
  Mat N;
  Vec z_j;
};

#endif
