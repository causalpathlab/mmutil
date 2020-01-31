#include "mmutil_aggregate_col.hh"

void
print_help(const char* fname) {

  const char* _desc =
      "Aggregate columns to create multiple types of summary stats.\n"
      "\n"
      "[Arguments]\n"
      "MTX        : Matrix Market file <i> <j> <value>\n"
      "COL        : Columns in the matrix <j>\n"
      "MEMBERSHIP : Membership for each column <j> <k>\n"
      "OUTPUT     : ${OUTPUT}.sum.gz ${OUTPUT}.mean.gz ${OUTPUT}.cv.gz \n"
      "\n";
  std::cerr << _desc << std::endl;
  std::cerr << fname << " MATCH MEMBERSHIP OUTPUT" << std::endl;
  std::cerr << std::endl;
}

int
main(const int argc, const char* argv[]) {

  if (argc != 5) {
    print_help(argv[0]);
    return EXIT_FAILURE;
  }

  const std::string mtx_file(argv[1]);
  const std::string col_file(argv[2]);
  const std::string membership_file(argv[3]);
  const std::string output(argv[4]);

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

  TLOG("Done");
  return EXIT_SUCCESS;
}
