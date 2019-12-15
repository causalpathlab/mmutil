#include "mmutil.hh"
#include "mmutil_merge_col.hh"

void print_help(const char* fname) {
  std::cerr << "Merge the columns of sparse matrices matching rows" << std::endl;
  std::cerr << std::endl;
  std::cerr << fname << " global_row count_threshold output" << std::endl;
  std::cerr << " { mtx[1] row[1] column[1] mtx[2] row[2] column[2] ... }" << std::endl;
  std::cerr << std::endl;
  std::cerr << "global_row      : A file that contains the names of rows." << std::endl;
  std::cerr << "count_threshold : The minimum number of non-zero elements per column" << std::endl;
  std::cerr << "output          : Header string for the output fileset." << std::endl;
  std::cerr << "mtx[i]          : i-th matrix market format file" << std::endl;
  std::cerr << "row[i]          : i-th row file" << std::endl;
  std::cerr << "column[i]       : i-th column file" << std::endl;
  std::cerr << std::endl;
}

int main(const int argc, const char* argv[]) {

  if (argc < 7) {
    print_help(argv[0]);
    return EXIT_FAILURE;
  }

  using Str = std::string;

  const Str glob_row_file(argv[1]);
  const Index column_threshold = boost::lexical_cast<Scalar>(argv[2]);
  const Str output(argv[3]);
  const int num_batches = (argc - 4) / 3;

  TLOG("Number of batches to merge: " << num_batches << " (argc: " << argc << ")");

  std::vector<std::string> mtx_files;
  std::vector<std::string> row_files;
  std::vector<std::string> col_files;

  for (int batch_index; batch_index < num_batches; ++batch_index) {
    int b = batch_index * 3 + 4;
    mtx_files.push_back(Str(argv[b]));
    row_files.push_back(Str(argv[b + 1]));
    col_files.push_back(Str(argv[b + 2]));
  }

  return run_merge_col(glob_row_file, column_threshold, output, mtx_files, row_files, col_files);
}
