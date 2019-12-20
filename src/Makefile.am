bin_PROGRAMS = mmutil_merge_col \
	mmutil_filter_row \
	mmutil_filter_col \
	mmutil_distribute_col \
	mmutil_rand_col	\
	mmutil_spectral_col

math = svd.hh

common = mmutil.hh mmutil_stat.hh \
        io.hh io_visitor.hh \
	utils/gzstream.hh utils/gzstream.cc \
	utils/util.hh utils/strbuf.hh

hnswlib = ext/hnswlib/bruteforce.h \
	ext/hnswlib/hnswalg.h \
	ext/hnswlib/hnswlib.h \
	ext/hnswlib/space_ip.h \
	ext/hnswlib/space_l2.h \
	ext/hnswlib/visited_list_pool.h

mmutil_merge_col_SOURCES = mmutil_run_merge_col.cc mmutil_merge_col.hh $(common)

mmutil_filter_row_SOURCES = mmutil_filter_row.cc $(common)

mmutil_filter_col_SOURCES = mmutil_filter_col.cc $(common)

mmutil_distribute_col_SOURCES = mmutil_distribute_col.cc $(common)

mmutil_spectral_col_SOURCES = mmutil_spectral_col.cc mmutil_spectral.hh $(common) $(math) 

mmutil_rand_col_SOURCES = mmutil_rand_col.cc $(common)
