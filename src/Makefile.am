bin_PROGRAMS = mmutil_merge_col \
    mmutil_split_col \
    mmutil_filter_row \
    mmutil_filter_col \
    mmutil_select_row \
    mmutil_select_col \
    mmutil_aggregate_col \
    mmutil_rand_col \
    mmutil_rand_conv_col \
    mmutil_normalize_col \
    mmutil_spectral_col \
    mmutil_match_col \
    mmutil_annotate_col \
    mmutil_embed_annot \
    mmutil_spectral_match_col \
    mmutil_spectral_cluster_col

math = \
    svd.hh \
    inference/adam.hh

rbm = mmutil_rbm.hh \
    inference/sampler.hh \
    inference/adam.hh

sgd = \
    inference/sgd_representation_gaussian.hh \
    inference/sgd_regression.hh

common = mmutil.hh \
    mmutil_stat.hh \
    eigen_util.hh \
    utils/sse.h \
    utils/cast.h \
    utils/math.hh \
    utils/fastexp.h \
    utils/fastlog.h \
    utils/fastgamma.h \
    std_util.hh \
    io.hh eigen_io.hh io_visitor.hh \
    utils/gzstream.hh \
    utils/gzstream.cc \
    utils/util.hh \
    utils/check.hh \
    utils/tuple_util.hh \
    utils/progress.hh \
    utils/strbuf.hh

hnswlib = ext/hnswlib/bruteforce.h \
    ext/hnswlib/hnswalg.h \
    ext/hnswlib/hnswlib.h \
    ext/hnswlib/space_ip.h \
    ext/hnswlib/space_l2.h \
    ext/hnswlib/visited_list_pool.h

#################
# merge / split #
#################

mmutil_merge_col_SOURCES = mmutil_run_merge_col.cc \
    mmutil_merge_col.hh $(common)

mmutil_split_col_SOURCES = mmutil_split_col.cc \
    mmutil_split_col.hh \
    $(common)

##############
# Annotation #
##############

mmutil_annotate_col_SOURCES = \
        mmutil_annotate_col.hh \
        inference/sampler.hh \
	mmutil_normalize.hh \
	mmutil_annotate_col.cc \
        $(common)

mmutil_embed_annot_SOURCES = \
	mmutil_normalize.hh \
	mmutil_spectral.hh \
	mmutil_embedding.hh \
	mmutil_annotate_embedding.hh \
	mmutil_annotate_embedding.cc \
	$(math) \
        $(common)

#####################
# filter and select #
#####################

mmutil_filter_row_SOURCES = mmutil_filter_row.cc \
    mmutil_filter_row.hh $(common)

mmutil_filter_col_SOURCES = mmutil_filter_col.cc \
    mmutil_filter_col.hh $(common)

mmutil_select_row_SOURCES = mmutil_select_row.cc \
    mmutil_select.hh $(common)

mmutil_select_col_SOURCES = mmutil_select_col.cc \
    mmutil_select.hh $(common)

mmutil_aggregate_col_SOURCES = mmutil_aggregate_col.cc \
    mmutil_aggregate_col.hh \
    $(common)

mmutil_spectral_col_SOURCES = mmutil_spectral_col.cc \
    mmutil_normalize.hh \
    mmutil_spectral.hh \
    $(common) \
    $(math)

mmutil_rand_col_SOURCES = mmutil_rand_col.cc \
    mmutil_rand_col.hh $(common)

mmutil_rand_conv_col_SOURCES = mmutil_rand_conv_col.cc \
    mmutil_rand_conv_col.hh $(common)

mmutil_normalize_col_SOURCES = mmutil_normalize_col.cc \
    mmutil_normalize.hh $(common)

mmutil_match_col_SOURCES = mmutil_match_col.cc \
    mmutil_match.hh $(common) $(hnslib)

mmutil_propagate_membership_SOURCES = mmutil_propagate_membership.cc \
    mmutil_propagate_membership.hh $(common)

mmutil_spectral_match_col_SOURCES = mmutil_spectral_match_col.cc \
    mmutil_match.hh $(common) $(hnslib) $(math)

mmutil_spectral_cluster_col_SOURCES = \
    mmutil_spectral_cluster_col.hh \
    mmutil_spectral_cluster_col.cc \
    mmutil_match.hh \
    mmutil_spectral.hh \
    mmutil_cluster.hh \
    inference/dpm.hh \
    inference/sampler.hh \
    inference/component_gaussian.hh \
    $(common) \
    $(math)
