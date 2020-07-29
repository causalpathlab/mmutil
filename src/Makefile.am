bin_PROGRAMS = mmutil_merge_col \
    mmutil_index_col \
    mmutil_filter_row \
    mmutil_filter_col \
    mmutil_select_row \
    mmutil_select_col \
    mmutil_aggregate_col \
    mmutil_normalize_col \
    mmutil_spectral_col \
    mmutil_match_col \
    mmutil_annotate_col \
    mmutil_doublet_qc \
    mmutil_embed_annot \
    mmutil_spectral_match_col \
    mmutil_spectral_cluster_col

math = \
    svd.hh \
    inference/adam.hh

rbm = mmutil_rbm.hh \
    inference/sampler.hh \
    inference/adam.hh

common = mmutil.hh                  \
    mmutil_util.hh                  \
    mmutil_stat.hh                  \
    mmutil_bgzf_util.hh             \
    mmutil_index.hh                 \
    eigen_util.hh                   \
    utils/sse.h                     \
    utils/cast.h                    \
    utils/math.hh                   \
    utils/fastexp.h                 \
    utils/fastlog.h                 \
    utils/fastgamma.h               \
    std_util.hh                     \
    io.hh eigen_io.hh io_visitor.hh \
    utils/gzstream.hh               \
    utils/gzstream.cc               \
    utils/util.hh                   \
    utils/check.hh                  \
    utils/tuple_util.hh             \
    utils/progress.hh               \
    utils/strbuf.hh                 \
    ext/tabix/bgzf.h                \
    ext/tabix/bgzf.c                \
    utils/bgzstream.hh              \
    utils/bgzstream.cc              \
    ext/tabix/kstring.c             \
    ext/tabix/kstring.h

hnswlib = ext/hnswlib/bruteforce.h \
    ext/hnswlib/hnswalg.h \
    ext/hnswlib/hnswlib.h \
    ext/hnswlib/space_ip.h \
    ext/hnswlib/space_l2.h \
    ext/hnswlib/visited_list_pool.h

############
# indexing #
############

mmutil_index_col_SOURCES = \
    mmutil_index_col.cc    \
    $(common)

#########
# merge #
#########

mmutil_merge_col_SOURCES = mmutil_run_merge_col.cc \
    mmutil_merge_col.hh $(common)

###########
# Doublet #
###########

mmutil_doublet_qc_SOURCES = \
	mmutil_normalize.hh \
	mmutil_match.hh \
	mmutil_spectral.hh \
	mmutil_doublet_qc.hh \
	mmutil_doublet_qc.cc \
        inference/sampler.hh \
	link_community.hh \
        $(common)

####################
# latent embedding #
####################

mmutil_spectral_col_SOURCES = \
    mmutil_spectral_col.cc \
    mmutil_normalize.hh \
    mmutil_spectral.hh \
    $(common) \
    $(math)


##############
# Annotation #
##############

mmutil_annotate_col_SOURCES = \
	mmutil_annotate.hh    \
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
    mmutil_aggregator.hh \
    mmutil_aggregate_col.hh \
    mmutil_spectral.hh \
    mmutil_match.hh \
    inference/sampler.hh \
    $(common)

mmutil_rand_col_SOURCES = mmutil_rand_col.cc \
    mmutil_rand_col.hh $(common)

mmutil_rand_conv_col_SOURCES = mmutil_rand_conv_col.cc \
    mmutil_rand_conv_col.hh $(common)

mmutil_normalize_col_SOURCES = mmutil_normalize_col.cc \
    mmutil_normalize.hh $(common)

#########
# match #
#########

mmutil_match_col_SOURCES = mmutil_match_col.cc \
    mmutil_match.hh $(common) $(hnslib)

mmutil_spectral_match_col_SOURCES = mmutil_spectral_match_col.cc \
    mmutil_match.hh $(common) $(hnslib) $(math)


##############
# clustering #
##############

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
