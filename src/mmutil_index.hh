#include "io.hh"
#include "mmutil.hh"
#include "mmutil_bgzf_util.hh"
#include "ext/tabix/bgzf.h"

#include <unordered_map>

/////////////////////////////////////////
// Matrix market column index format:  //
// column_index <space> memory_address //
/////////////////////////////////////////

#ifndef MMUTIL_INDEX_HH_
#define MMUTIL_INDEX_HH_

/// @param bgz_file : bgzipped mtx file
/// @param idx_file : index file for the bgz file
int build_mmutil_index(std::string bgz_file, std::string idx_file);

using idx_pair_t = std::tuple<Index, Index>;

/// @param idx_file : index file for the bgz file
/// @param idx      : index map (a vector of idx_pair_t)
int read_mmutil_index(std::string idx_file, std::vector<idx_pair_t> &idx);

/// @param mtx_file : bgzipped mtx file
/// @param idx_file : index file for the bgz file
/// @param subcol   : subset of column indexes (zero-based)
/// @return sparse matrix
template <typename VEC>
SpMat read_eigen_sparse_subset_col(std::string mtx_file,
                                   std::string idx_file,
                                   const VEC &subcol);

////////////////////////////////////////////////////////////////

struct mm_column_indexer_t {

    explicit mm_column_indexer_t()
    {
        first_off = 0;
        last_off = 0;
        lineno = 0;
        last_col = 0;
        fp_set = false;
        col2file.clear();
    }

    void set_file(BGZF *_fp)
    {
        fp = _fp;
        fp_set = true;
    }

    void eval_after_header(Index max_row, Index max_col, Index max_nnz)
    {
        ASSERT(fp_set, "BGZF file pointer must be set");
#ifdef DEBUG
        TLOG("#Rows: " << max_row << ", #Cols: " << max_col
                       << ", #NZ: " << max_nnz);
#endif
        last_col = 0; // coordinate index
        first_off = last_off = bgzf_tell(fp);
    }

    void eval(Index row, Index col, Scalar weight)
    {

        if (lineno == 0) {  // first column position &
            last_col = col; // offset are already found
            col2file.emplace_back(std::make_tuple(col, first_off));
        }

        if (col != last_col) { // the previous one was a change point

            ASSERT(col > last_col, "MTX must be sorted by columns");

            Index save_off = bgzf_tell(fp);
            ASSERT(save_off >= last_off, "corrupted");
            col2file.emplace_back(std::make_tuple(col, last_off));
            last_col = col; // prepare for the next
        }
        last_off = bgzf_tell(fp);

        ++lineno;
    }

    void eval_end_of_file()
    {
        fp_set = false;
#ifdef DEBUG
        TLOG("Finished reading the file: " << lineno << " lines");
#endif
    }

    BGZF *fp; // little unsafe
    Index first_off;
    Index last_off;
    Index lineno;
    Index last_col;

    using map_t = std::vector<std::tuple<Index, Index>>;

    const map_t &operator()() const { return col2file; }

private:
    map_t col2file;
    bool fp_set;
};

///////////////////////////////////////
// index bgzipped matrix market file //
///////////////////////////////////////

int build_mmutil_index(std::string bgz_file,        // bgzip file
                       std::string index_file = "") // index file
{

    if (index_file.length() == 0) {
        index_file = bgz_file + ".index";
    }

    if (bgzf_is_bgzf(bgz_file.c_str()) != 1) {
        ELOG("This file is not bgzipped: " << bgz_file);
        return EXIT_FAILURE;
    }

    if (file_exists(index_file)) {
        WLOG("Index file exists: " << index_file);
        return EXIT_SUCCESS;
    }

    mm_column_indexer_t indexer;
    CHECK(visit_bgzf(bgz_file, indexer));

    ogzstream ofs(index_file.c_str(), std::ios::out);
    write_tuple_stream(ofs, indexer());
    ofs.close();

    return EXIT_SUCCESS;
}

int
read_mmutil_index(std::string index_file, std::vector<idx_pair_t> &_index)
{
    _index.clear();
    igzstream ifs(index_file.c_str(), std::ios::in);
    int ret = read_pair_stream(ifs, _index);
    ifs.close();

    auto less_op = [](idx_pair_t &lhs, idx_pair_t &rhs) {
        return std::get<0>(lhs) < std::get<0>(rhs);
    };

    std::sort(_index.begin(), _index.end(), less_op);

    const Index N = _index.size();
    if (N < 1)
        return EXIT_FAILURE;
    return ret;
}

struct memory_block_t {
    Index lb;
    Index lb_mem;
    Index ub;
    Index ub_mem;
};

std::vector<memory_block_t>
find_consecutive_blocks(const std::vector<idx_pair_t> &index,
                        const std::vector<Index> &subcol)
{

    const Index N = index.size();
    ASSERT(N > 1, "Empty index map");

    std::unordered_set<Index> _subset;
    _subset.reserve(subcol.size());
    std::copy(subcol.begin(),
              subcol.end(),
              std::inserter(_subset, _subset.end()));

    const Index gap = 100;

#ifdef DEBUG
    TLOG("Built a subset of " << _subset.size() << " columns");
#endif

    std::vector<idx_pair_t> blocks;
    bool in_frag = false;
    Index beg = 0, end = 0;
    for (Index j = 0; j < N; ++j) {
        const Index i = std::get<0>(index[j]);
        if (_subset.count(i) > 0) {
            if (!in_frag) {     // beginning of the block
                in_frag = true; //
                beg = j;        // from this j
            }
        } else if (in_frag) {
            end = j; // finish the block here
            in_frag = false;
            blocks.emplace_back(std::make_tuple(beg, end));
        }
    }

    if (in_frag) {
        blocks.emplace_back(std::make_tuple(beg, N));
    }

#ifdef DEBUG
    TLOG("Identified " << blocks.size() << " block(s)");
#endif
    std::vector<memory_block_t> ret;

    for (auto b : blocks) {
        Index lb, lb_mem, ub = N, ub_mem = 0;
        std::tie(lb, lb_mem) = index[std::get<0>(b)];

        if (std::get<1>(b) < (N - 1)) {
            std::tie(ub, ub_mem) = index[std::get<1>(b)];
        }

#ifdef DEBUG
        TLOG("Block [" << lb << ", " << ub << ")");
#endif

        ret.emplace_back(memory_block_t{ lb, lb_mem, ub, ub_mem });
    }
    return ret;
}

template <typename VEC>
SpMat
read_eigen_sparse_subset_col(std::string mtx_file,
                             std::string index_file,
                             const VEC &subcol)
{

    using _reader_t = eigen_triplet_reader_remapped_cols_t;
    using Index = _reader_t::index_t;

    std::vector<idx_pair_t> index_tab;
#ifdef DEBUG
    TLOG("Reading the index file: " << index_file);
#endif

    CHECK(read_mmutil_index(index_file, index_tab));

#ifdef DEBUG
    TLOG("Read the file: " << index_tab.size() << " elements");
#endif

    const auto blocks = find_consecutive_blocks(index_tab, subcol);

#ifdef DEBUG
    TLOG("Found " << blocks.size() << " blocks");
#endif

    _reader_t::TripletVec Tvec; // keep accumulating this

    Index max_col = 0;
    Index max_row = 0;
    for (auto block : blocks) {
        _reader_t::index_map_t remap;
        for (Index j = block.lb; j < block.ub; ++j) {
            remap[j] = max_col;
            ++max_col;
        }
        _reader_t reader(Tvec, remap);
        CHECK(peek_bgzf_header(mtx_file, reader));
        max_row = std::max(max_row, reader.max_row);
        CHECK(visit_bgzf_block(mtx_file, block.lb_mem, block.ub_mem, reader));
    }

    SpMat X(max_row, max_col);
    X.reserve(Tvec.size());
    X.setFromTriplets(Tvec.begin(), Tvec.end());

#ifdef DEBUG
    TLOG("Constructed a sparse matrix with m = " << X.nonZeros());
#endif

    return X;
}

#endif
