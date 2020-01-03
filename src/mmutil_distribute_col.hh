#include "mmutil.hh"
#include "mmutil_stat.hh"

#ifndef MMUTIL_DISTRIBUTE_COL_HH_
#define MMUTIL_DISTRIBUTE_COL_HH_

void
distribute_col(const std::string mtx_file,         // matrix market
               const std::string membership_file,  // membership
               const std::string output) {         //

  using Str = std::string;

  std::unordered_map<Str, Str> _column_membership;
  CHECK(read_pair_file(membership_file, _column_membership));

  // Filter out zero count columns
  col_stat_collector_t collector;
  visit_matrix_market_file(mtx_file, collector);
  const Index max_row = collector.max_row, max_col = collector.max_col;
  ASSERT(_column_membership.size() >= max_col, "Insufficient number of columns");

  std::vector<Scalar> nnz_col(max_col);
  std_vector(collector.Col_N, nnz_col);

  std::vector<Str> column_names;
  std::transform(_column_membership.begin(),        //
                 _column_membership.end(),          //
                 std::back_inserter(column_names),  //
                 [](auto& pp) -> Str { return pp.first; });

  TLOG("Calculated the number of non-zero elements of the columns");

  using IdxVec = std::vector<Index>;
  using ColPtr = std::shared_ptr<IdxVec>;
  std::unordered_map<Str, ColPtr> distributed_columns;

  Index k = 0;
  for (auto pp : _column_membership) {

    const Str key = std::get<1>(pp);

    if (distributed_columns.count(key) < 1) {
      ColPtr val(new IdxVec);
      distributed_columns[key] = val;
    }

    IdxVec& old_indexes = *(distributed_columns[key].get());
    old_indexes.push_back(k);

    if (++k >= max_col) break;  // stop over-assignment
  }

  using copier_t = triplet_copier_remapped_cols_t<Index, Scalar>;
  using remap_t  = copier_t::index_map_t;

  for (auto pp : distributed_columns) {
    TLOG("Adding a new batch = " << pp.first);

    std::vector<Str> out_column_names;
    const IdxVec& old_indexes = *(pp.second.get());

    Index j = 0;
    remap_t remap;
    Index nnz = 0;
    for (auto i : old_indexes) {
      if (nnz_col.at(i) > 0) {
        remap[i] = j;
        out_column_names.push_back(column_names.at(i));
        nnz += static_cast<Index>(nnz_col.at(i));
        j++;
      }
    }

    if (j < 1) continue;

    TLOG("New batch [" << pp.first << "] [NNZ = " << nnz << "]");

    const Str output_mtx_file    = output + "_" + (pp.first) + ".mtx.gz";
    const Str output_column_file = output + "_" + (pp.first) + ".columns.gz";

    write_vector_file(output_column_file, out_column_names);

    copier_t copier(output_mtx_file, remap, nnz);
    visit_matrix_market_file(mtx_file, copier);
  }
}

#endif
