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

namespace mmutil { namespace index {

using namespace mmutil::bgzf;

/**
   @param bgz_file : bgzipped mtx file
   @param idx_file : index file for the bgz file
*/
int build_mmutil_index(std::string bgz_file, std::string idx_file);

/**
   @param idx_file : index file for the bgz file
   @param idx      : index map (a vector of memory locations)
*/
int read_mmutil_index(std::string idx_file, std::vector<Index> &idx);

/**
   @param mtx_file : bgzipped mtx file
   @param idx_file : index file for the bgz file
   @param subcol   : subset of column indexes (zero-based)
   @return sparse matrix
*/
template <typename VEC>
SpMat read_eigen_sparse_subset_col(std::string mtx_file,
                                   std::string idx_file,
                                   const VEC &subcol);

/**
   @param mtx_file : bgzipped mtx file
   @param idx_tab  : a vector of indexing pairs
   @param subcol   : subset of column indexes (zero-based)
   @return sparse matrix
*/
template <typename VEC>
SpMat read_eigen_sparse_subset_col(std::string mtx_file,
                                   std::vector<Index> &index_tab,
                                   const VEC &subcol);

/**
   @param mtx_file : bgzipped mtx file
   @param idx_tab  : a vector of indexing pairs
   @param subrow   : subset of row indexes (zero-based)
   @param subcol   : subset of column indexes (zero-based)
   @return sparse matrix
*/
template <typename VEC>
SpMat read_eigen_sparse_subset_row_col(std::string mtx_file,
                                       std::vector<Index> &index_tab,
                                       const VEC &subrow,
                                       const VEC &subcol);

/**
   @param mtx_file : bgzipped mtx file
   @param idx_tab  : a vector of indexing pairs
   @param subrow   : subset of row indexes (zero-based)
   @param subcol   : subset of column indexes (zero-based)
   @return sparse matrix
*/
template <typename VEC>
SpMat read_eigen_sparse_subset_row_col(std::string mtx_file,
                                       std::string index_file,
                                       const VEC &subrow,
                                       const VEC &subcol);

/**
   @param mtx_file matrix market file
   @param index_tab a vector of index pairs
*/
int check_index_tab(std::string mtx_file, std::vector<Index> &index_tab);

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
        TLOG("#Rows: " << max_row << ", #Cols: " << max_col
                       << ", #NZ: " << max_nnz);
        last_col = 0; // coordinate index
        first_off = last_off = bgzf_tell(fp);
        col2file.reserve(max_col);
    }

    void eval(Index row, Index col, Scalar weight)
    {

        if (lineno == 0) {  // first column position &
            last_col = col; // offset are already found
            col2file.emplace_back(std::make_tuple(col, first_off));
        }

        if (col != last_col) { // the last one was a change point

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
        TLOG("Finished indexing the file of " << lineno << " lines");
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

int build_mmutil_index(std::string mtx_file,        // bgzip file
                       std::string index_file = "") // index file
{

    if (index_file.length() == 0) {
        index_file = mtx_file + ".index";
    }

    if (bgzf_is_bgzf(mtx_file.c_str()) != 1) {
        ELOG("This file is not bgzipped: " << mtx_file);
        return EXIT_FAILURE;
    }

    if (file_exists(index_file)) {
        WLOG("Index file exists: " << index_file);
        return EXIT_SUCCESS;
    }

    mm_info_reader_t info;
    CHECK(peek_bgzf_header(mtx_file, info));

    mm_column_indexer_t indexer;
    CHECK(visit_bgzf(mtx_file, indexer));

    const mm_column_indexer_t::map_t &_map = indexer();

    //////////////////////////////////////////
    // Check if we can find all the columns //
    //////////////////////////////////////////

    TLOG("Check the index size: " << _map.size() << " vs. " << info.max_col);

    const Index sz = _map.size();
    const Index last_col = sz > 0 ? std::get<0>(_map[sz - 1]) : 0;

    if (last_col != (info.max_col - 1)) {
        ELOG("Failed to index all the columns: " << last_col << " < "
                                                 << (info.max_col - 1));
        return EXIT_FAILURE;
    }

    TLOG("Writing " << index_file << "...");

    ogzstream ofs(index_file.c_str(), std::ios::out);
    write_tuple_stream(ofs, _map);
    ofs.close();

    TLOG("Built the file: " << index_file);

    return EXIT_SUCCESS;
}

int
read_mmutil_index(std::string index_file, std::vector<Index> &_index)
{
    _index.clear();
    std::vector<std::tuple<Index, Index>> temp;
    igzstream ifs(index_file.c_str(), std::ios::in);
    int ret = read_pair_stream(ifs, temp);
    ifs.close();

    const Index N = temp.size();

    if (N < 1)
        return EXIT_FAILURE;

    Index MaxIdx = 0;
    for (auto pp : temp) {
        MaxIdx = std::max(std::get<0>(pp), MaxIdx);
    }

    // Fill in missing locations
    _index.resize(MaxIdx + 1);
    std::fill(std::begin(_index), std::end(_index), MISSING_POS);

    for (auto pp : temp) {
        _index[std::get<0>(pp)] = std::get<1>(pp);
    }

    // Update missing spots with the next one
    for (Index j = 0; j < (MaxIdx - 1); ++j) {
        if (_index[j] == MISSING_POS)
            _index[j] = _index[j + 1];
    }

    return ret;
}

struct memory_block_t {
    Index lb;
    Index lb_mem;
    Index ub;
    Index ub_mem;
};

std::vector<memory_block_t>
find_consecutive_blocks(const std::vector<Index> &index_tab,
                        const std::vector<Index> &subcol,
                        const Index gap = 10)
{

    const Index N = index_tab.size();
    ASSERT(N > 1, "Empty index map");

    std::vector<Index> sorted(subcol.size());
    std::copy(subcol.begin(), subcol.end(), sorted.begin());
    std::sort(sorted.begin(), sorted.end());

    std::vector<std::tuple<Index, Index>> intervals;
    {
        Index beg = sorted[0];
        Index end = beg;

        for (Index jj = 1; jj < sorted.size(); ++jj) {
            const Index ii = sorted[jj];
            if (ii >= (end + gap)) {                  //
                intervals.emplace_back(beg, end + 1); //
                beg = ii;                             // start a new interval
                end = ii;                             //
            } else {                                  // extend this interval
                end = ii;                             //
            }
        }

        if (beg <= sorted[sorted.size() - 1]) {
            intervals.emplace_back(beg, end + 1);
        }
    }

    std::vector<memory_block_t> ret;

    for (auto intv : intervals) {

        Index lb, lb_mem, ub, ub_mem = 0;
        std::tie(lb, ub) = intv;

        if (lb >= N)
            continue;

        lb_mem = index_tab[lb];

        // if (ub == lb) {
        //     ub = ub + 1;
        // }

        if (ub < N) {
            ub_mem = index_tab[ub];
        }

        // TLOG(lb << ", " << ub << " " << lb_mem << " " << ub_mem);
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
    std::vector<Index> index_tab;
    CHECK(read_mmutil_index(index_file, index_tab));
    CHECK(check_index_tab(mtx_file, index_tab));
    return read_eigen_sparse_subset_col(mtx_file, index_tab, subcol);
}

/**
   @param mtx_file matrix market file
   @param index_tab a vector of index pairs
*/
int
check_index_tab(std::string mtx_file, std::vector<Index> &index_tab)
{

    mm_info_reader_t info;
    CHECK(peek_bgzf_header(mtx_file, info));

    const Index sz = index_tab.size();
    const Index last_col = sz - 1;

    if (last_col == (info.max_col - 1))
        return EXIT_SUCCESS;

    ELOG("This index file is corrupted: " << last_col << " < "
                                          << (info.max_col - 1));
    return EXIT_FAILURE;
}

template <typename VEC>
SpMat
read_eigen_sparse_subset_col(std::string mtx_file,
                             std::vector<Index> &index_tab,
                             const VEC &subcol)
{

    mm_info_reader_t info;
    CHECK(peek_bgzf_header(mtx_file, info));

    using _reader_t = eigen_triplet_reader_remapped_cols_t;
    using Index = _reader_t::index_t;
#ifdef DEBUG
    CHECK(check_index_tab(mtx_file, index_tab));
#endif

    Index max_col = 0;                   // Make sure that
    _reader_t::index_map_t subcol_order; // we keep the same order
    for (auto k : subcol) {              // of subcol
        subcol_order[k] = max_col++;
    }

    const auto blocks = find_consecutive_blocks(index_tab, subcol);

    _reader_t::TripletVec Tvec; // keep accumulating this
    Index max_row = info.max_row;
    for (auto block : blocks) {
        _reader_t::index_map_t loc_map;
        for (Index j = block.lb; j < block.ub; ++j) {
            loc_map[j] = subcol_order[j];
        }
        _reader_t reader(Tvec, loc_map);

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

template <typename VEC>
SpMat
read_eigen_sparse_subset_row_col(std::string mtx_file,
                                 std::string index_file,
                                 const VEC &subrow,
                                 const VEC &subcol)
{

    std::vector<Index> index_tab;
    CHECK(read_mmutil_index(index_file, index_tab));
    return read_eigen_sparse_subset_row_col(mtx_file,
                                            index_tab,
                                            subrow,
                                            subcol);
}

template <typename VEC>
SpMat
read_eigen_sparse_subset_row_col(std::string mtx_file,
                                 std::vector<Index> &index_tab,
                                 const VEC &subrow,
                                 const VEC &subcol)
{

    using _reader_t = eigen_triplet_reader_remapped_rows_cols_t;
    using Index = _reader_t::index_t;

    Index max_col = 0;                   // Make sure that
    _reader_t::index_map_t subcol_order; // we keep the same order
    for (auto k : subcol) {              // of subcol
        subcol_order[k] = max_col++;
    }

    const auto blocks = find_consecutive_blocks(index_tab, subcol);

    _reader_t::index_map_t remap_row;
    for (Index new_index = 0; new_index < subrow.size(); ++new_index) {
        const Index old_index = subrow.at(new_index);
        remap_row[old_index] = new_index;
    }

    _reader_t::TripletVec Tvec; // keep accumulating this

    Index max_row = subrow.size();
    for (auto block : blocks) {
        _reader_t::index_map_t remap_col;
        for (Index old_index = block.lb; old_index < block.ub; ++old_index) {
            remap_col[old_index] = subcol_order[old_index];
        }
        _reader_t reader(Tvec, remap_row, remap_col);
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

} // namespace index
} // namespace mmutil
#endif
