////////////////////////////////////////////////////////////////
// I/O routines
#include <cctype>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Sparse>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "io_visitor.hh"
#include "utils/gzstream.hh"
#include "utils/strbuf.hh"
#include "utils/util.hh"

#ifndef UTIL_IO_HH_
#define UTIL_IO_HH_

/////////////////////////////////
// common utility for data I/O //
/////////////////////////////////
bool is_file_gz(const std::string filename) {
  if (filename.size() < 3) return false;
  return filename.substr(filename.size() - 3) == ".gz";
}

std::shared_ptr<std::ifstream> open_ifstream(const std::string filename) {
  std::shared_ptr<std::ifstream> ret(new std::ifstream(filename.c_str(), std::ios::in));
  return ret;
}

std::shared_ptr<igzstream> open_igzstream(const std::string filename) {
  std::shared_ptr<igzstream> ret(new igzstream(filename.c_str(), std::ios::in));
  return ret;
}

template <typename IFS, typename T1, typename T2>
auto read_pair_stream(IFS &ifs, std::unordered_map<T1, T2> &in) {
  in.clear();
  T1 v;
  T2 w;
  while (ifs >> v >> w) {
    in[v] = w;
  }
  ERR_RET(in.size() == 0, "empty file");
  return EXIT_SUCCESS;
}

template <typename T1, typename T2>
auto read_pair_file(const std::string filename, std::unordered_map<T1, T2> &in) {
  auto ret = EXIT_SUCCESS;

  if (is_file_gz(filename)) {
    igzstream ifs(filename.c_str(), std::ios::in);
    ret = read_pair_stream(ifs, in);
    ifs.close();
  } else {
    std::ifstream ifs(filename.c_str(), std::ios::in);
    ret = read_pair_stream(ifs, in);
    ifs.close();
  }
  return ret;
}

template <typename IFS, typename T>
auto read_vector_stream(IFS &ifs, std::vector<T> &in) {
  in.clear();
  T v;
  while (ifs >> v) {
    in.push_back(v);
  }
  ERR_RET(in.size() == 0, "empty vector");
  return EXIT_SUCCESS;
}

template <typename T>
auto read_vector_file(const std::string filename, std::vector<T> &in) {
  auto ret = EXIT_SUCCESS;

  if (is_file_gz(filename)) {
    igzstream ifs(filename.c_str(), std::ios::in);
    ret = read_vector_stream(ifs, in);
    ifs.close();
  } else {
    std::ifstream ifs(filename.c_str(), std::ios::in);
    ret = read_vector_stream(ifs, in);
    ifs.close();
  }
  return ret;
}

/////////////////////////////////////////////////////////////
// read matrix market triplets and construct sparse matrix //
/////////////////////////////////////////////////////////////

struct TripletReader {

  using scalar_t = float;
  using index_t = std::ptrdiff_t;

  using Triplet = std::tuple<index_t, index_t, scalar_t>;
  using TripletVec = std::vector<Triplet>;

  explicit TripletReader(TripletVec &_tvec) : Tvec(_tvec) {
    max_row = 0;
    max_col = 0;
    max_elem = 0;
    Tvec.clear();
    TLOG("Start reading a list of triplets");
  }

  void set_dimension(const index_t r, const index_t c, const index_t e) {
    max_row = r;
    max_col = c;
    max_elem = e;
  }

  void eval(const index_t row, const index_t col, const scalar_t weight) {
    Tvec.push_back(Triplet(row, col, weight));
  }

  void eval_end() {
    if (Tvec.size() < max_elem) {
      WLOG("This file may have lost elements : " << Tvec.size() << " vs. " << max_elem);
    }
    TLOG("Finished reading a list of triplets");
  }

  index_t max_row;
  index_t max_col;
  index_t max_elem;
  TripletVec &Tvec;
};

template <typename IFS>
inline auto read_matrix_market_stream(IFS &ifs) {
  TripletReader::TripletVec Tvec;

  TripletReader reader(Tvec);

  visit_matrix_market_stream(ifs, reader);

  auto max_row = reader.max_row;
  auto max_col = reader.max_col;

  return std::make_tuple(Tvec, max_row, max_col);
}

auto read_matrix_market_file(const std::string filename) {

  TripletReader::TripletVec Tvec;
  TripletReader reader(Tvec);

  visit_matrix_market_file(filename, reader);

  auto max_row = reader.max_row;
  auto max_col = reader.max_col;

  return std::make_tuple(Tvec, max_row, max_col);
}

///////////////////
// simple writer //
///////////////////

template <typename OFS, typename Derived>
void write_matrix_market_stream(OFS &ofs, const Eigen::SparseMatrixBase<Derived> &out) {
  ofs.precision(4);

  const Derived &M = out.derived();

  ofs << "%%MatrixMarket matrix coordinate integer general" << std::endl;
  ofs << M.rows() << " " << M.cols() << " " << M.nonZeros() << std::endl;

  const typename Derived::Index INTERVAL = 1e6;
  const typename Derived::Index max_triples = M.nonZeros();
  using Index = typename Derived::Index;
  Index _num_triples = 0;

  // column major
  for (auto k = 0; k < M.outerSize(); ++k) {
    for (typename Derived::InnerIterator it(M, k); it; ++it) {
      const Index i = it.row() + 1;  // fix zero-based to one-based
      const Index j = it.col() + 1;  // fix zero-based to one-based
      const auto v = it.value();
      ofs << i << " " << j << " " << v << std::endl;

      if (++_num_triples % INTERVAL == 0) {
        std::cerr << "\r" << std::left << std::setfill('.') << std::setw(30) << "Writing "
                  << std::right << std::setfill(' ') << std::setw(10) << (_num_triples / INTERVAL)
                  << " x 1M triplets (total " << std::setw(10) << (max_triples / INTERVAL) << ")"
                  << std::flush;
      }
    }
  }
  std::cerr << std::endl;
}

template <typename T>
void write_matrix_market_file(const std::string filename, T &out) {
  if (is_file_gz(filename)) {
    ogzstream ofs(filename.c_str(), std::ios::out);
    write_matrix_market_stream(ofs, out);
    ofs.close();
  } else {
    std::ofstream ofs(filename.c_str(), std::ios::out);
    write_matrix_market_stream(ofs, out);
    ofs.close();
  }
}

/////////////////////////////////////
// frequently used output routines //
/////////////////////////////////////

template <typename OFS, typename Vec>
void write_pair_stream(OFS &ofs, const Vec &vec) {
  ofs.precision(4);

  for (auto pp : vec) {
    ofs << std::get<0>(pp) << " " << std::get<1>(pp) << std::endl;
  }
}

template <typename Vec>
void write_pair_file(const std::string filename, const Vec &out) {
  if (is_file_gz(filename)) {
    ogzstream ofs(filename.c_str(), std::ios::out);
    write_pair_stream(ofs, out);
    ofs.close();
  } else {
    std::ofstream ofs(filename.c_str(), std::ios::out);
    write_pair_stream(ofs, out);
    ofs.close();
  }
}

template <typename OFS, typename Vec>
void write_vector_stream(OFS &ofs, const Vec &vec) {
  ofs.precision(4);

  for (auto pp : vec) {
    ofs << pp << std::endl;
  }
}

template <typename Vec>
void write_vector_file(const std::string filename, const Vec &out) {
  if (is_file_gz(filename)) {
    ogzstream ofs(filename.c_str(), std::ios::out);
    write_vector_stream(ofs, out);
    ofs.close();
  } else {
    std::ofstream ofs(filename.c_str(), std::ios::out);
    write_vector_stream(ofs, out);
    ofs.close();
  }
}

/////////////////////////////
// identify dimensionality //
/////////////////////////////

template <typename IFS>
auto num_cols(IFS &ifs) {
  std::istreambuf_iterator<char> eos;
  std::istreambuf_iterator<char> it(ifs);
  const auto eol = '\n';

  auto ret = 1;
  for (; it != eos && *it != eol; ++it) {
    char c = *it;
    if (isspace(c) && c != eol) ++ret;
  }

  return ret;
}

template <typename IFS>
auto num_rows(IFS &ifs) {
  std::istreambuf_iterator<char> eos;
  std::istreambuf_iterator<char> it(ifs);
  const char eol = '\n';

  auto ret = 0;
  for (; it != eos; ++it)
    if (*it == eol) ++ret;

  return ret;
}

template <typename IFS, typename T>
auto read_data_stream(IFS &ifs, T &in) {
  typedef typename T::Scalar elem_t;

  typedef enum _state_t { S_WORD, S_EOW, S_EOL } state_t;
  const auto eol = '\n';
  std::istreambuf_iterator<char> END;
  std::istreambuf_iterator<char> it(ifs);

  std::vector<elem_t> data;
  strbuf_t strbuf;
  state_t state = S_EOL;

  auto nr = 0u;  // number of rows
  auto nc = 1u;  // number of columns

  elem_t val;
  auto nmissing = 0;

  for (; it != END; ++it) {
    char c = *it;

    if (c == eol) {
      if (state == S_WORD) {
        val = strbuf.lexical_cast<elem_t>();

        if (!isfinite(val)) nmissing++;

        data.push_back(val);
        strbuf.clear();
      } else if (state == S_EOW) {
        data.push_back(NAN);
        nmissing++;
      }
      state = S_EOL;
      nr++;
    } else if (isspace(c)) {
      if (state == S_WORD) {
        val = strbuf.lexical_cast<elem_t>();

        if (!isfinite(val)) nmissing++;

        data.push_back(val);
        strbuf.clear();
      } else {
        data.push_back(NAN);
        nmissing++;
      }
      state = S_EOW;
      if (nr == 0) nc++;

    } else {
      strbuf.add(c);
      state = S_WORD;
    }
  }

#ifdef DEBUG
  TLOG("Found " << nmissing << " missing values");
#endif

  auto mtot = data.size();
  ERR_RET(mtot != (nr * nc),
          "# data points: " << mtot << " elements in " << nr << " x " << nc << " matrix");
  ERR_RET(mtot < 1, "empty file");
  ERR_RET(nr < 1, "zero number of rows; incomplete line?");
  in = Eigen::Map<T>(data.data(), nc, nr);
  in.transposeInPlace();

  return EXIT_SUCCESS;
}

////////////////////////////////////////////////////////////////
template <typename T>
auto read_data_file(const std::string filename, T &in) {
  auto ret = EXIT_SUCCESS;

  if (is_file_gz(filename)) {
    igzstream ifs(filename.c_str(), std::ios::in);
    ret = read_data_stream(ifs, in);
    ifs.close();
  } else {
    std::ifstream ifs(filename.c_str(), std::ios::in);
    ret = read_data_stream(ifs, in);
    ifs.close();
  }

  return ret;
}

////////////////////////////////////////////////////////////////
template <typename T>
auto read_data_file(const std::string filename) {
  typename std::shared_ptr<T> ret(new T{});
  auto &in = *ret.get();

  if (is_file_gz(filename)) {
    igzstream ifs(filename.c_str(), std::ios::in);
    CHK_ERR_EXIT(read_data_stream(ifs, in), "Failed to read " << filename);
    ifs.close();
  } else {
    std::ifstream ifs(filename.c_str(), std::ios::in);
    CHK_ERR_EXIT(read_data_stream(ifs, in), "Failed to read " << filename);
    ifs.close();
  }

  return ret;
}

////////////////////////////////////////////////////////////////
template <typename OFS, typename Derived>
void write_data_stream(OFS &ofs, const Eigen::MatrixBase<Derived> &out) {
  ofs.precision(4);

  const Derived &M = out.derived();

  for (auto r = 0u; r < M.rows(); ++r) {
    ofs << M.coeff(r, 0);
    for (auto c = 1u; c < M.cols(); ++c) ofs << " " << M.coeff(r, c);
    ofs << std::endl;
  }
}

template <typename OFS, typename Derived>
void write_data_stream(OFS &ofs, const Eigen::SparseMatrixBase<Derived> &out) {
  ofs.precision(4);

  const Derived &M = out.derived();
  using Index = typename Derived::Index;
  using Scalar = typename Derived::Scalar;

  // Not necessarily column major
  const Index INTERVAL = 1000;
  const Index max_outer = M.outerSize();
  const Index MAX_PRINT = (max_outer / INTERVAL);

  for (auto k = 0; k < max_outer; ++k) {
    for (typename Derived::InnerIterator it(M, k); it; ++it) {
      const Index i = it.row();
      const Index j = it.col();
      const Scalar v = it.value();
      ofs << i << " " << j << " " << v << std::endl;
    }
    if ((k + 1) % INTERVAL == 0) {
      std::cerr << "\rWriting " << std::right << std::setfill(' ') << std::setw(10)
                << (k / INTERVAL);
      std::cerr << " x 1k outer-iterations (total ";
      std::cerr << std::setw(10) << MAX_PRINT << ")\r" << std::flush;
    }
  }
  std::cerr << std::endl;
}

////////////////////////////////////////////////////////////////
template <typename T>
void write_data_file(const std::string filename, const T &out) {
  if (is_file_gz(filename)) {
    ogzstream ofs(filename.c_str(), std::ios::out);
    write_data_stream(ofs, out);
    ofs.close();
  } else {
    std::ofstream ofs(filename.c_str(), std::ios::out);
    write_data_stream(ofs, out);
    ofs.close();
  }
}

#endif
