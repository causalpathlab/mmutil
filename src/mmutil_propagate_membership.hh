#include "mmutil.hh"
#include "mmutil_stat.hh"

#ifndef MMUTIL_COLLAPSE_MATCH_HH_
#define MMUTIL_COLLAPSE_MATCH_HH_

auto
propagate_membership(const std::string match_file, //
                     const std::string membership_file, //
                     const Scalar decay)
{
    ///////////////////////////
    // read matching results //
    ///////////////////////////

    eigen_io::row_name_vec_t::type i_name;
    eigen_io::col_name_vec_t::type j_name;
    eigen_io::row_index_map_t::type i_index;
    eigen_io::col_index_map_t::type j_index;

    SpMat M;
    eigen_io::read_named_eigen_sparse_file(match_file, //
                                           eigen_io::row_name_vec_t(i_name), //
                                           eigen_io::col_name_vec_t(j_name), //
                                           eigen_io::row_index_map_t(
                                               i_index), //
                                           eigen_io::col_index_map_t(
                                               j_index), //
                                           M);

    TLOG("Read the matching matrix: " << M.rows() << " x " << M.cols());

    ////////////////////////////////
    // read the membership matrix //
    ////////////////////////////////

    eigen_io::col_name_vec_t::type k_name;
    eigen_io::col_index_map_t::type k_index;

    SpMat Z;

    read_named_membership_file(membership_file,
                               eigen_io::row_index_map_t(j_index), //
                               eigen_io::col_name_vec_t(k_name), //
                               eigen_io::col_index_map_t(k_index), //
                               Z);

    auto d2w = [&decay](const Scalar &x) -> Scalar {
        return std::exp(-decay * x);
    };

    Mat deg = M.unaryExpr(d2w) * Mat::Ones(M.cols(), 1);
    Scalar denom = std::max(deg.maxCoeff(), static_cast<Scalar>(1.));
    SpMat WZ = (M.unaryExpr(d2w) * Z) / denom;

    //////////////////////////////
    // Find argmax for each row //
    //////////////////////////////

    using out_tup = std::tuple<std::string, std::string, Scalar>;
    std::vector<out_tup> out_vec;
    out_vec.reserve(WZ.rows());

    ///////////////////////
    // must be row-major //
    ///////////////////////

    for (Index i = 0; i < WZ.outerSize(); ++i) {
        Index argmax = 0;
        Scalar maxval = 0;

        for (SpMat::InnerIterator ct(WZ, i); ct; ++ct) {
            const Index k = ct.col();
            const Scalar wz_ik = ct.value();

            if (wz_ik > maxval) {
                maxval = wz_ik;
                argmax = k;
            }
        }

        out_vec.push_back(
            std::make_tuple(i_name.at(i), k_name.at(argmax), maxval));
    }

    return out_vec;
}

#endif
