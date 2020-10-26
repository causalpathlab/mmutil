bin_PROGRAMS = mmutil_select_row mmutil_select_col mmutil_filter_row mmutil_filter_col mmutil_merge_col mmutil_index_col mmutil_row_score mmutil_col_score mmutil_histogram mmutil_bbknn mmutil_cfa_col mmutil_aggregate_col mmutil_spectral_col mmutil_match_col mmutil_annotate_col mmutil_embed_annot mmutil_spectral_match_col mmutil_spectral_cluster_col mmutil_diff mmutil_simulate

tabix = ext/tabix/bgzf.h ext/tabix/bgzf.c utils/bgzstream.hh utils/bgzstream.cc ext/tabix/kstring.c ext/tabix/kstring.h

utils = utils/sse.h utils/cast.h utils/math.hh  utils/stat.hh utils/fastexp.h utils/fastlog.h utils/fastgamma.h utils/gzstream.hh utils/gzstream.cc utils/util.hh utils/check.hh utils/tuple_util.hh utils/progress.hh utils/strbuf.hh utils/eigen_util.hh utils/std_util.hh

common = mmutil.hh mmutil_util.hh mmutil_stat.hh mmutil_bgzf_util.hh mmutil_index.hh mmutil_io.hh io.hh io_visitor.hh $(utils) $(tabix)

hnswlib = ext/hnswlib/bruteforce.h ext/hnswlib/hnswalg.h ext/hnswlib/hnswlib.h ext/hnswlib/space_ip.h ext/hnswlib/space_l2.h ext/hnswlib/visited_list_pool.h

spectral = svd.hh mmutil_spectral.hh inference/adam.hh

############
# indexing #
############

mmutil_index_col_SOURCES = mmutil_index_col.cc $(common)

#########
# merge #
#########

mmutil_merge_col_SOURCES = mmutil_run_merge_col.cc mmutil_merge_col.hh $(common)

#######################
# batch-balancing knn #
#######################

mmutil_bbknn_SOURCES = mmutil_normalize.hh mmutil_match.hh mmutil_bbknn.hh mmutil_bbknn.cc $(common) $(spectral)

####################
# latent embedding #
####################

mmutil_spectral_col_SOURCES = mmutil_spectral_col.cc mmutil_normalize.hh $(common) $(spectral)

##############
# Annotation #
##############

mmutil_annotate_col_SOURCES = mmutil_annotate.hh mmutil_annotate_col.hh inference/sampler.hh mmutil_normalize.hh mmutil_annotate_col.cc $(common)

mmutil_embed_annot_SOURCES = mmutil_normalize.hh mmutil_embedding.hh mmutil_annotate_embedding.hh mmutil_annotate_embedding.cc $(spectral) $(common)

#####################
# filter and select #
#####################

mmutil_filter_row_SOURCES = mmutil_filter_row.cc mmutil_filter_row.hh $(common)

mmutil_row_score_SOURCES = mmutil_row_score.cc mmutil_score.hh mmutil_filter_row.hh $(common)

mmutil_filter_col_SOURCES = mmutil_filter_col.cc mmutil_filter_col.hh $(common)

mmutil_col_score_SOURCES = mmutil_col_score.cc mmutil_score.hh mmutil_filter_col.hh $(common)

mmutil_select_row_SOURCES = mmutil_select_row.cc mmutil_select.hh $(common)

mmutil_select_col_SOURCES = mmutil_select_col.cc mmutil_select.hh $(common)

mmutil_histogram_SOURCES = mmutil_histogram.cc mmutil_histogram.hh $(common)

###############
# aggregation #
###############

mmutil_aggregate_col_SOURCES = mmutil_aggregate_col.cc mmutil_aggregate_col.hh mmutil_pois.hh $(common)

mmutil_simulate_SOURCES = mmutil_simulate.cc mmutil_simulate.hh $(common)

###########################
# counterfactual analysis #
###########################

mmutil_cfa_col_SOURCES = mmutil_cfa_col.cc mmutil_pois.hh mmutil_cfa_col.hh mmutil_match.hh $(common)

mmutil_diff_SOURCES = mmutil_diff.cc mmutil_pois.hh mmutil_diff.hh $(common)

#########
# match #
#########

mmutil_match_col_SOURCES = mmutil_match_col.cc mmutil_match.hh $(common) $(hnslib)

mmutil_spectral_match_col_SOURCES = mmutil_spectral_match_col.cc mmutil_match.hh $(common) $(hnslib) $(spectral)

##############
# clustering #
##############

mmutil_spectral_cluster_col_SOURCES = mmutil_spectral_cluster_col.hh mmutil_spectral_cluster_col.cc mmutil_match.hh mmutil_cluster.hh inference/dpm.hh inference/sampler.hh inference/component_gaussian.hh $(common) $(spectral)

