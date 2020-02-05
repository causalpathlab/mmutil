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

    S1.setZero();
    S2.setZero();
    N.setZero();

    // to check empty columns
    Vec _nnz_col = Z.transpose() * Mat::Ones(Z.rows(), 1);
    std_vector(_nnz_col, nnz_col);

    TLOG("Start aggregating the statistics: " << kk << " x " << D);
  }

  void eval(const index_t i, const index_t j, const scalar_t x_ij) {
    if (j < nn && j >= 0 && nnz_col.at(j) > 0) {
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

 private:
  std::vector<Index> nnz_col;
};

void
aggregate_col(const std::string mtx_file,         //
              const std::string col_file,         //
              const std::string membership_file,  //
              const std::string output) {

  ///////////////////////
  // read column names //
  ///////////////////////

  std::vector<std::string> columns;
  read_vector_file(col_file, columns);

  eigen_io::row_index_map_t::type j_index;
  j_index.reserve(columns.size());
  Index j = 0;
  for (auto s : columns) {
    j_index[s] = j++;
  }

  ///////////////////////////////
  // match with the membership //
  ///////////////////////////////

  eigen_io::col_name_vec_t::type k_name;
  eigen_io::col_index_map_t::type k_index;

  SpMat Zsparse;

  read_named_membership_file(membership_file,
                             eigen_io::row_index_map_t(j_index),  //
                             eigen_io::col_name_vec_t(k_name),    //
                             eigen_io::col_index_map_t(k_index),  //
                             Zsparse);

  if (Zsparse.rows() == 0 || Zsparse.cols() == 0) {
    WLOG("Empty latent membership matrix");
    return;
  }

  Mat Z = Zsparse;       //
  Z.transposeInPlace();  // cluster x sample

  //////////////////////////////////////
  // collect statistics from the data //
  //////////////////////////////////////

  TLOG("Might take some time... (but memory-efficient)");

  mat_stat_collector_t collector(Z);
  visit_matrix_market_file(mtx_file, collector);

  {
    TLOG("Writing S1 stats");
    Mat S1 = collector.S1.transpose();
    write_data_file(output + ".s1.gz", S1);
  }

  {
    TLOG("Writing S2 stats");
    Mat S2 = collector.S2.transpose();
    write_data_file(output + ".s2.gz", S2);
  }

  {
    TLOG("Writing N stats");
    Mat N = collector.N.transpose();
    write_data_file(output + ".n.gz", N);
  }

  write_vector_file(output + ".columns.gz", k_name);
}

#endif
