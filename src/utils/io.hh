////////////////////////////////////////////////////////////////
// I/O routines
#include <cctype>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "utils/gzstream.hh"
#include "utils/strbuf.hh"

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Sparse>

#ifndef UTIL_IO_HH_
#define UTIL_IO_HH_

/////////////////////////////////
// common utility for data I/O //
/////////////////////////////////
auto is_file_gz(const std::string filename) {
  if (filename.size() < 3) return false;
  return filename.substr(filename.size() - 3) == ".gz";
}

std::shared_ptr<std::ifstream> open_ifstream(const std::string filename) {
  std::shared_ptr<std::ifstream> ret(
      new std::ifstream(filename.c_str(), std::ios::in));
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
auto read_pair_file(const std::string filename,
                    std::unordered_map<T1, T2> &in) {
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

template <typename IFS>
auto read_matrix_market_stream(IFS &ifs) {
  ///////////////////////
  // basic definitions //
  ///////////////////////

  using Scalar = float;
  using Index = long int;

  //////////////////////////
  // Finite state machine //
  //////////////////////////

  typedef enum _state_t { S_COMMENT, S_WORD, S_EOW, S_EOL } state_t;
  const auto eol = '\n';
  const auto comment = '%';

  std::istreambuf_iterator<char> END;
  std::istreambuf_iterator<char> it(ifs);

  strbuf_t strbuf;
  state_t state = S_EOL;

  auto num_rows = 0u;  // number of rows
  auto num_cols = 0u;  // number of columns

  // read the first line and headers
  // %%MatrixMarket matrix coordinate integer general

  std::vector<Index> Dims(3);

  auto extract_idx_word = [&]() {
    const Index _idx = strbuf.get<Index>();
    if (num_cols < Dims.size()) {
      Dims[num_cols] = _idx;
    }
    state = S_EOW;
    strbuf.clear();
    return _idx;
  };

  for (; num_rows < 1 && it != END; ++it) {
    char c = *it;

    // Skip the comment line. It doesn't count toward the line
    // count, and we don't bother reading the content.
    if (state == S_COMMENT) {
      if (c == eol) state = S_EOL;
      continue;
    }

    if (c == comment) {
      state = S_COMMENT;
      continue;
    }

    // Do the regular parsing of a triplet

    if (c == eol) {
      if (state == S_WORD) {
        const auto nelem = extract_idx_word();
        num_cols++;
#ifdef DEBUG
        TLOG("Elements : " << nelem);
#endif
      }

      state = S_EOL;
      num_rows++;

    } else if (isspace(c)) {
      auto d = extract_idx_word();
      num_cols++;
#ifdef DEBUG
      TLOG("Dimsension : " << d);
#endif
    } else {
      strbuf.add(c);
      state = S_WORD;
    }
  }

#ifdef DEBUG
  for (auto d : Dims) {
    TLOG(d);
  }
  TLOG("debug -- read the header.");
#endif

  // Read a list of triplets
  TLOG("Start reading a list of triplets");

  Index row, col;
  Scalar weight;

  using Triplet = std::tuple<Index, Index, Scalar>;
  using TripletVec = std::vector<Triplet>;

  TripletVec Tvec;

  auto read_triplet = [&]() {
    switch (num_cols) {
      case 0:
        // row = strbuf.get<Index>();
        row = strbuf.take_int();
        break;
      case 1:
        // col = strbuf.get<Index>();
        col = strbuf.take_int();
        break;
      case 2:
        // weight = strbuf.get<Scalar>();
        weight = strbuf.take_float();
        break;
    }
    state = S_EOW;
    strbuf.clear();
  };

  num_cols = 0;
  num_rows = 0;

  const Index max_row = Dims[0];
  const Index max_col = Dims[1];
  const Index max_elem = Dims[2];
  const Index INTERVAL = 1e6;

  for (; num_rows < max_elem && it != END; ++it) {
    char c = *it;

    // Skip the comment line. It doesn't count toward the line
    // count, and we don't bother reading the content.
    if (state == S_COMMENT) {
      if (c == eol) state = S_EOL;
      continue;
    }

    if (c == comment) {
      state = S_COMMENT;
      continue;
    }

    // Do the regular parsing of a triplet

    if (c == eol) {
      if (state == S_WORD) {
        read_triplet();
        num_cols++;
      }

      state = S_EOL;
      num_rows++;

      if (row < 0 || row > max_row)
        WLOG("Ignore unexpected row" << std::setw(10) << row);
      if (col < 0 || col > max_col)
        WLOG("Ignore unexpected column" << std::setw(10) << col);

      Tvec.push_back(Triplet(row - 1, col - 1, weight));
      num_cols = 0;

      if (num_rows % INTERVAL == 0) {
        std::cerr << "\r" << std::setw(30) << "Reading " << std::setw(10)
                  << (num_rows / INTERVAL) << " x 1M triplets (total "
                  << std::setw(10) << (max_elem / INTERVAL) << ")"
                  << std::flush;
      }
    } else if (isspace(c) && strbuf.size() > 0) {
      read_triplet();
      num_cols++;
    } else {
      strbuf.add(c);
      state = S_WORD;
    }
  }
  std::cerr << std::endl;

  TLOG("Finished reading a list of triplets");

  if (Tvec.size() < max_elem) {
    WLOG("This file may have lost elements : " << Tvec.size() << " vs. "
                                               << max_elem);
  }

  return std::make_tuple(Tvec, max_row, max_col);
}

auto read_matrix_market_file(const std::string filename) {
  if (is_file_gz(filename)) {
    igzstream ifs(filename.c_str(), std::ios::in);
    auto ret = read_matrix_market_stream(ifs);
    ifs.close();
    return ret;
  } else {
    std::ifstream ifs(filename.c_str(), std::ios::in);
    auto ret = read_matrix_market_stream(ifs);
    ifs.close();
    return ret;
  }
}

template <typename OFS, typename Derived>
void write_matrix_market_stream(OFS &ofs,
                                const Eigen::SparseMatrixBase<Derived> &out) {
  ofs.precision(4);

  const Derived &M = out.derived();

  ofs << "%%MatrixMarket matrix coordinate integer general" << std::endl;
  ofs << M.rows() << " " << M.cols() << " " << M.nonZeros() << std::endl;

  const typename Derived::Index INTERVAL = 1e6;
  const typename Derived::Index max_triples = M.nonZeros();
  typename Derived::Index _num_triples = 0;

  // column major
  for (auto k = 0; k < M.outerSize(); ++k) {
    for (typename Derived::InnerIterator it(M, k); it; ++it) {
      const auto i = it.row() + 1;  // fix zero-based to one-based
      const auto j = it.col() + 1;  // fix zero-based to one-based
      const auto v = it.value();
      ofs << i << " " << j << " " << v << std::endl;

      if (++_num_triples % INTERVAL == 0) {
        std::cerr << "\r" << std::setw(30) << "Writing " << std::setw(10)
                  << (_num_triples / INTERVAL) << " x 1M triplets (total "
                  << std::setw(10) << (max_triples / INTERVAL) << ")"
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
          "# data points: " << mtot << " elements in " << nr << " x " << nc
                            << " matrix");
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

  for (auto k = 0; k < max_outer; ++k) {
    for (typename Derived::InnerIterator it(M, k); it; ++it) {
      const Index i = it.row();
      const Index j = it.col();
      const Scalar v = it.value();
      ofs << i << " " << j << " " << v << std::endl;
    }
    if ((k + 1) % INTERVAL == 0) {
      std::cerr << "\rWriting " << std::setw(10) << (k / INTERVAL)
                << " x 1k outer-iterations (total " << std::setw(10)
                << (max_outer / INTERVAL) << ")" << std::flush;
    }
  }
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
