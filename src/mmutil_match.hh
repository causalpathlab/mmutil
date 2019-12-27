#include <unordered_set>

#include "eigen_util.hh"
#include "ext/hnswlib/hnswlib.h"
#include "mmutil.hh"
#include "mmutil_stat.hh"
#include "utils/progress.hh"

#ifndef MMUTIL_MATCH_HH_
#define MMUTIL_MATCH_HH_

/////////////////////////////////
// k-nearest neighbor matching //
/////////////////////////////////

using KnnAlg = hnswlib::HierarchicalNSW<Scalar>;

struct SrcSparseRowsT {
  explicit SrcSparseRowsT(const SpMat& _data) : data(_data){};
  const SpMat& data;
};

struct TgtSparseRowsT {
  explicit TgtSparseRowsT(const SpMat& _data) : data(_data){};
  const SpMat& data;
};

struct KNN {
  explicit KNN(const std::string _val) : val(std::stoi(_val)) {}
  const std::size_t val;
};

struct BILINKS {
  explicit BILINKS(const std::string _val) : val(std::stoi(_val)) {}
  const std::size_t val;
};

struct NNLIST {
  explicit NNLIST(const std::string _val) : val(std::stoi(_val)) {}
  const std::size_t val;
};

using index_triplet_vec = std::vector<std::tuple<Index, Index, Scalar> >;

///////////////////////////////////////////
// search over the rows of sparse matrix //
///////////////////////////////////////////

int search_knn(const SrcSparseRowsT _SrcRows,  //
               const TgtSparseRowsT _TgtRows,  //
               const KNN _knn,                 //
               const BILINKS _bilinks,         //
               const NNLIST _nnlist,           //
               index_triplet_vec& out);

////////////////////////////////
// search over the dense data //
////////////////////////////////

struct SrcDataT {
  explicit SrcDataT(const float* _data, const Index d, const Index s)
      : data(_data), vecdim(d), vecsize(s) {}
  const float* data;
  const Index vecdim;
  const Index vecsize;
};

struct TgtDataT {
  explicit TgtDataT(const float* _data, const Index d, const Index s)
      : data(_data), vecdim(d), vecsize(s) {}
  const float* data;
  const Index vecdim;
  const Index vecsize;
};

int search_knn(const SrcDataT _SrcData,  //
               const TgtDataT _TgtData,  //
               const KNN _knn,           //
               const BILINKS _bilinks,   //
               const NNLIST _nnlist,     //
               index_triplet_vec& out);

////////////////////////////////////////////////////////////////

int search_knn(const SrcSparseRowsT _SrcRows,  //
               const TgtSparseRowsT _TgtRows,  //
               const KNN _knn,                 //
               const BILINKS _bilinks,         //
               const NNLIST _nnlist,           //
               index_triplet_vec& out) {

  const SpMat& SrcRows = _SrcRows.data;
  const SpMat& TgtRows = _TgtRows.data;

  ERR_RET(TgtRows.cols() != SrcRows.cols(),
          "Target and source data must have the same dimensionality");

  const std::size_t knn           = _knn.val;
  const std::size_t param_bilinks = _bilinks.val;
  const std::size_t param_nnlist  = _nnlist.val;

  const std::size_t vecdim  = TgtRows.cols();
  const std::size_t vecsize = TgtRows.rows();

  ERR_RET(param_bilinks >= vecdim, "too big M value");
  ERR_RET(param_bilinks < 2, "too small M value");
  ERR_RET(param_nnlist < knn, "too small N value");

  // Construct KnnAlg interface
  hnswlib::InnerProductSpace vecspace(vecdim);
  KnnAlg alg(&vecspace, vecsize, param_bilinks, param_nnlist);
  alg.ef_ = param_nnlist;

  TLOG("Initializing kNN algorithm");

  // We need to store target data in the memory (or in the disk)
  std::vector<float> target_data(TgtRows.rows() * TgtRows.cols());
  std::fill(target_data.begin(), target_data.end(), 0.0);
  {
    float* mass = target_data.data();

    progress_bar_t<Index> prog(vecsize, 1e3);

    for (Index i = 0; i < TgtRows.outerSize(); ++i) {

      float norm = 0.0;
      for (SpMat::InnerIterator it(TgtRows, i); it; ++it) {
        float w = it.value();
        norm += w * w;
      }
      norm = std::sqrt(std::max(norm, static_cast<float>(1.0)));

      for (SpMat::InnerIterator it(TgtRows, i); it; ++it) {
        const Index j               = it.col();
        target_data[vecdim * i + j] = it.value() / norm;
      }

      alg.addPoint((void*)(mass + vecdim * i), static_cast<std::size_t>(i));
      prog.update();
      prog(std::cerr);
    }
  }
  ////////////
  // recall //
  ////////////

  TLOG("Finding " << knn << " nearest neighbors");

  {
    progress_bar_t<Index> prog(SrcRows.outerSize(), 1e3);

    std::vector<float> lookup(vecdim);
    for (Index i = 0; i < SrcRows.outerSize(); ++i) {

      float norm = 0.0;
      std::fill(lookup.begin(), lookup.end(), 0.0);
      for (SpMat::InnerIterator it(SrcRows, i); it; ++it) {
        float w = it.value();
        norm += w * w;
      }
      norm = std::sqrt(std::max(norm, static_cast<float>(1.0)));

      for (SpMat::InnerIterator it(SrcRows, i); it; ++it) {
        float w          = it.value();
        lookup[it.col()] = w / norm;
      }

      auto pq = alg.searchKnn((void*)lookup.data(), knn);
      float d = 0;
      std::size_t j;
      while (!pq.empty()) {
        std::tie(d, j) = pq.top();
        out.push_back(std::make_tuple(i, j, d));
        pq.pop();
      }
      prog.update();
      prog(std::cerr);
    }
  }
  TLOG("Done kNN searches");
  return EXIT_SUCCESS;
}

int search_knn(const SrcDataT _SrcData,  //
               const TgtDataT _TgtData,  //
               const KNN _knn,           //
               const BILINKS _bilinks,   //
               const NNLIST _nnlist,     //
               index_triplet_vec& out) {

  ERR_RET(_SrcData.vecdim == _TgtData.vecdim,
          "source and target must have the same dimensionality");

  const std::size_t knn           = _knn.val;
  const std::size_t param_bilinks = _bilinks.val;
  const std::size_t param_nnlist  = _nnlist.val;

  const std::size_t vecdim  = _TgtData.vecdim;
  const std::size_t vecsize = _TgtData.vecsize;

  ERR_RET(param_bilinks >= vecdim, "too big M value");
  ERR_RET(param_bilinks < 2, "too small M value");
  ERR_RET(param_nnlist < knn, "too small N value");

  // Construct KnnAlg interface
  hnswlib::InnerProductSpace vecspace(vecdim);
  KnnAlg alg(&vecspace, vecsize, param_bilinks, param_nnlist);
  alg.ef_ = param_nnlist;

  TLOG("Initializing kNN algorithm");

  {
    const float* mass = _TgtData.data;

    progress_bar_t<Index> prog(vecsize, 1e3);

    for (Index i = 0; i < vecsize; ++i) {
      alg.addPoint((void*)(mass + vecdim * i), static_cast<std::size_t>(i));
      prog.update();
      prog(std::cerr);
    }
  }

  ////////////
  // recall //
  ////////////

  TLOG("Finding " << knn << " nearest neighbors");

  {
    const float* mass = _SrcData.data;
    progress_bar_t<Index> prog(_SrcData.vecsize, 1e3);

    for (Index i = 0; i < _SrcData.vecsize; ++i) {
      auto pq = alg.searchKnn((void*)(mass + vecdim * i), knn);
      float d = 0;
      std::size_t j;
      while (!pq.empty()) {
        std::tie(d, j) = pq.top();
        out.push_back(std::make_tuple(i, j, d));
        pq.pop();
      }
      prog.update();
      prog(std::cerr);
    }
  }
  TLOG("Done kNN searches");
  return EXIT_SUCCESS;
}

#endif
