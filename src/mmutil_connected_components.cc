#include "mmutil.hh"

void print_help(const char* fname) {
  std::cerr
      << "Identify connected components and create a file for each component"
      << std::endl;
  std::cerr << std::endl;
  std::cerr << fname << " cutoff mtx_file output" << std::endl;
  std::cerr << std::endl;
}

int main(const int argc, const char* argv[]) {
  if (argc < 4) {
    print_help(argv[0]);
    return EXIT_FAILURE;
  }

  using Scalar = float;
  using Index = long int;
  using SpMat = Eigen::SparseMatrix<Scalar>;
  using Str = std::string;

  const Scalar jaccard_cutoff = boost::lexical_cast<Scalar>(argv[1]);
  const Str mtx_file(argv[2]);
  const Str output(argv[3]);

  ///////////////////
  // Read the data //
  ///////////////////

  using Triplet = std::tuple<Index, Index, Scalar>;
  using TripletVec = std::vector<Triplet>;
  TripletVec Tvec;
  Index max_row, max_col;
  std::tie(Tvec, max_row, max_col) = read_matrix_market_file(mtx_file);

  TLOG(max_row << " x " << max_col);

  /////////////////////////////////////////////////////////////
  // construct boost graph and identify connected components //
  /////////////////////////////////////////////////////////////

  using namespace boost;

  using Graph =
      adjacency_list<vecS, vecS, undirectedS, no_property, no_property>;

  using Vertex = typename graph_traits<Graph>::vertex_descriptor;
  using Edge = typename Graph::edge_descriptor;

  Graph G;
  const Index max_vertex = max_row + max_col;

  for (Index u = num_vertices(G); u < max_vertex; ++u) add_vertex(G);

  for (auto tt : Tvec) {
    Index r, c;
    Scalar w;
    bool has_edge;
    Edge e;

    std::tie(r, c, w) = tt;

    Index u = r, v = c + max_row;

    tie(e, has_edge) = edge(u, v, G);
    if (!has_edge) add_edge(u, v, G);
  }

  /////////////////////////////////////////
  // construct shared neighborhood graph //
  /////////////////////////////////////////

  Graph S;

  Graph::adjacency_iterator ri, rEnd;
  Graph::adjacency_iterator ci, cEnd;
  std::unordered_map<Index, int> sn_count;

  auto _vertex = [&](const auto& c) {
    // take column and convert it to vertex
    return static_cast<Index>(c + max_row);
  };

  auto _column = [&](const auto& v) {
    // take vertex and convert it to column
    return static_cast<Vertex>(v - max_row);
  };

  auto _jaccard = [](const auto n12, const auto N1, const auto N2) {
    const Scalar _one = 1.0;
    const Scalar denom = std::max(static_cast<Scalar>(N1 + N2 - n12), _one);
    return static_cast<Scalar>(n12) / denom;
  };

  for (Vertex v = max_row; v < num_vertices(G); ++v) {
    Graph::adjacency_iterator ri, re;
    Graph::adjacency_iterator ci, ce;
    sn_count.clear();

    const Scalar N1 = degree(v, G);
    const Index this_col = _column(v);

    for (tie(ri, re) = adjacent_vertices(v, G); ri != re; ++ri) {
      for (tie(ci, ce) = adjacent_vertices(*ri, G); ci != ce; ++ci) {
        const Index other_col = _column(*ci);
        if (other_col <= this_col) continue;
        const int prev =
            (sn_count.count(other_col) > 0) ? sn_count.at(other_col) : 0;
        sn_count[other_col]++;
      }
    }

    for (auto it : sn_count) {
      const Index other_col = it.first;
      const Scalar N2 = degree(_vertex(other_col), G);
      const Scalar jaccard = _jaccard(it.second, N1, N2);

      if (jaccard >= jaccard_cutoff) {
        add_edge(this_col, other_col, S);
        // std::cout << this_col << " " << other_col << std::endl;
      }
    }
  }

  // Graph::edge_iterator ei, eEnd;
  // for(tie(ei, eEnd) = edges(S); ei!=eEnd; ++ei) {
  //   auto u =  source(*ei, S);
  //   auto v =  target(*ei, S);
  //   std::cout << u << " " << v << std::endl;
  // }

  using IndexVec = std::vector<Index>;
  IndexVec membership(num_vertices(S));
  const Index numComp = connected_components(S, &membership[0]);
  TLOG("Found " << numComp << " connected components");

  // TODO: output each component's mtx file



  return EXIT_SUCCESS;
}
