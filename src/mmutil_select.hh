#include "mmutil.hh"
#include "mmutil_stat.hh"

#ifndef MMUTIL_SELECT_HH_
#define MMUTIL_SELECT_HH_

void
copy_selected_columns(const std::string mtx_file, //
                      const std::string full_column_file, //
                      const std::string selected_column_file, //
                      const std::string output)
{
    using Str = std::string;
    using copier_t = triplet_copier_remapped_cols_t<Index, Scalar>;

    std::vector<Str> _selected(0);
    CHECK(read_vector_file(selected_column_file, _selected));
    std::unordered_set<Str> selected(_selected.begin(), _selected.end());

    std::vector<Str> full_column_names(0);
    CHECK(read_vector_file(full_column_file, full_column_names));

    col_stat_collector_t collector;
    visit_matrix_market_file(mtx_file, collector);
    const IntVec &nnz_col = collector.Col_N;
    const Index max_row = collector.max_row, max_col = collector.max_col;
    ASSERT(full_column_names.size() >= max_col,
           "Insufficient number of columns");

    std::vector<Index> cols(max_col);
    std::iota(std::begin(cols), std::end(cols), 0);
    std::vector<Index> valid_cols;
    auto _found = [&](const Index j) {
        return selected.count(full_column_names.at(j)) > 0;
    };
    std::copy_if(cols.begin(), cols.end(), std::back_inserter(valid_cols),
                 _found);

    TLOG("Found " << valid_cols.size() << " columns");

    copier_t::index_map_t remap;

    std::vector<Str> out_column_names;
    std::vector<Index> index_out(valid_cols.size());
    std::vector<Scalar> out_scores;
    Index i = 0;
    Index NNZ = 0;
    for (Index old_index : valid_cols) {
        remap[old_index] = i;
        out_column_names.push_back(full_column_names.at(old_index));

        NNZ += nnz_col(old_index);
        ++i;
    }

    TLOG("Created valid column names");

    const Str output_column_file = output + ".columns.gz";
    const Str output_mtx_file = output + ".mtx.gz";

    write_vector_file(output_column_file, out_column_names);

    copier_t copier(output_mtx_file, remap, NNZ);
    visit_matrix_market_file(mtx_file, copier);
}

#endif
