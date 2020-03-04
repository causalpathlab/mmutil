#include <random>
#include <string>

#include "io.hh"
#include "mmutil.hh"
#include "mmutil_stat.hh"

#ifndef MMUTIL_RANDOM_COL_HH_
#define MMUTIL_RANDOM_COL_HH_

void
copy_random_columns(const Index Nsample,         //
                    std::string mtx_file,        //
                    std::string column_file,     //
                    std::string output_mtx_file, //
                    std::string output_column_file)
{
    std::vector<std::string> column_names(0);
    CHECK(read_vector_file(column_file, column_names));

    col_stat_collector_t collector;
    visit_matrix_market_file(mtx_file, collector);
    const IntVec &nnz_col = collector.Col_N;

    std::random_device rd;
    std::mt19937 rgen(rd());

    std::vector<Index> index_r(column_names.size());
    std::iota(index_r.begin(), index_r.end(), 0);
    std::shuffle(index_r.begin(), index_r.end(), rgen);

    using copier_t = triplet_copier_remapped_cols_t<Index, Scalar>;
    copier_t::index_map_t remap;

    std::vector<std::string> out_column_names;
    Index NNZ = 0;

    for (Index new_index = 0;                                               //
         new_index < std::min(static_cast<Index>(index_r.size()), Nsample); //
         ++new_index) {
        const Index old_index = index_r.at(new_index);
        remap[old_index] = new_index;

        out_column_names.push_back(column_names.at(old_index));
        NNZ += nnz_col(old_index);
    }

    copier_t copier(output_mtx_file, remap, NNZ);
    visit_matrix_market_file(mtx_file, copier);

    write_vector_file(output_column_file, out_column_names);
}

#endif
