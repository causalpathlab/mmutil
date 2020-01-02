#include <random>

#include "mmutil_cluster.hh"

int main(const int argc, const char* argv[]) {

  Mat X, Ctrue;
  std::vector<Index> Ztrue;
  std::tie(X, Ztrue, Ctrue) = simulate_gaussian_mixture();

  for(auto k : Ztrue)      {
    std::cout << " "  << k;
  }
  std::cout << std::endl;

  std::cout << Ctrue.transpose() << std::endl;

  clustering_options_t options;

  options.K = 7;

  using F0 = trunc_dpm_t<Mat>;
  using F = multi_gaussian_component_t<Mat>;

  estimate_mixture_of_columns<F0, F>(X, options);

  return EXIT_SUCCESS;
}

// template<typename F>
// Scalar overall_score
