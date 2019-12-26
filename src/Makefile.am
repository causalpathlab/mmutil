bin_PROGRAMS = mmutil_merge_col \
    mmutil_filter_row \
    mmutil_filter_col \
    mmutil_select_row \
    mmutil_select_col \
    mmutil_distribute_col \
    mmutil_rand_col \
    mmutil_normalize_col \
    mmutil_spectral_col \
    mmutil_match_col

math = svd.hh

common = mmutil.hh mmutil_stat.hh \
    eigen_util.hh \
    std_util.hh \
    io.hh io_visitor.hh \
    utils/gzstream.hh utils/gzstream.cc \
    utils/util.hh utils/strbuf.hh

hnswlib = ext/hnswlib/bruteforce.h \
    ext/hnswlib/hnswalg.h \
    ext/hnswlib/hnswlib.h \
    ext/hnswlib/space_ip.h \
    ext/hnswlib/space_l2.h \
    ext/hnswlib/visited_list_pool.h

mmutil_merge_col_SOURCES = mmutil_run_merge_col.cc \
    mmutil_merge_col.hh $(common)

mmutil_filter_row_SOURCES = mmutil_filter_row.cc \
    mmutil_filter_row.hh $(common)

mmutil_filter_col_SOURCES = mmutil_filter_col.cc \
    mmutil_filter_col.hh $(common)

mmutil_select_row_SOURCES = mmutil_select_row.cc \
    mmutil_select.hh $(common)

mmutil_select_col_SOURCES = mmutil_select_col.cc \
    mmutil_select.hh $(common)

mmutil_distribute_col_SOURCES = mmutil_distribute_col.cc \
    mmutil_distribute_col.hh \
    $(common)

mmutil_spectral_col_SOURCES = mmutil_spectral_col.cc \
    mmutil_normalize.hh \
    mmutil_spectral.hh \
    $(common) \
    $(math)

mmutil_rand_col_SOURCES = mmutil_rand_col.cc \
    mmutil_rand_col.hh $(common)

mmutil_normalize_col_SOURCES = mmutil_normalize_col.cc \
    mmutil_normalize.hh $(common)

mmutil_match_col_SOURCES = mmutil_match_col.cc \
    mmutil_match.hh $(common) $(hnslib)
