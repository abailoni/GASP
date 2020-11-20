import time

import numpy as np
from nifty.graph import rag as nrag

from ..utils.graph import build_lifted_graph_from_rag, get_rag
from ..utils.various import check_offsets, find_indices_direct_neighbors_in_offsets

class AccumulatorLongRangeAffs(object):
    def __init__(self, offsets,
                 used_offsets=None,
                 offsets_weights=None,
                 verbose=True,
                 n_threads=-1,
                 invert_affinities=False,
                 statistic='mean',
                 offset_probabilities=None,
                 return_dict=False):

        offsets = check_offsets(offsets)

        # Parse inputs:
        if used_offsets is not None:
            assert len(used_offsets) < offsets.shape[0]
            if offset_probabilities is not None:
                offset_probabilities = np.require(offset_probabilities, dtype='float32')
                assert len(offset_probabilities) == len(offsets)
                offset_probabilities = offset_probabilities[used_offsets]
            if offsets_weights is not None:
                offsets_weights = np.require(offsets_weights, dtype='float32')
                assert len(offsets_weights) == len(offsets)
                offsets_weights = offsets_weights[used_offsets]

        self.offsets_probabilities = offset_probabilities
        self.used_offsets = used_offsets
        self.offsets_weights = offsets_weights


        self.used_offsets = used_offsets
        self.return_dict = return_dict
        self.statistic = statistic

        assert isinstance(n_threads, int)

        self.offsets = offsets
        self.verbose = verbose
        self.n_threads = n_threads
        self.invert_affinities = invert_affinities
        self.offset_probabilities = offset_probabilities

    def __call__(self, affinities, segmentation):
        tick = time.time()

        # Use only few channels from the affinities, if we are not using all offsets:
        offsets = self.offsets
        offsets_weights = self.offsets_weights
        if self.used_offsets is not None:
            assert len(self.used_offsets) < self.offsets.shape[0]
            offsets = self.offsets[self.used_offsets]
            affinities = affinities[self.used_offsets]
            if isinstance(offsets_weights, (list, tuple)):
                offsets_weights = np.array(offsets_weights)
            offsets_weights = offsets_weights[self.used_offsets]

        assert affinities.ndim == 4
        # affinities = affinities[:3]
        assert affinities.shape[0] == offsets.shape[0]

        if self.invert_affinities:
            affinities = 1. - affinities

        # Build rag and compute node sizes:
        if self.verbose:
            print("Computing rag...")
            tick = time.time()

        # If there was a label -1, now its value in the rag is given by the maximum label
        # (and it will be ignored later on)
        rag, has_background_label = get_rag(segmentation, self.n_threads)

        if self.verbose:
            print("Took {} s!".format(time.time() - tick))
            tick = time.time()

        out_dict = {}
        out_dict['rag'] = rag

        # Build graph including long-range connections:
        if self.verbose:
            print("Building (lifted) graph...")

        # -----------------------
        # Lifted edges:
        # -----------------------
        # Get rid of local offsets:
        is_direct_neigh_offset, indices_local_offsets = find_indices_direct_neighbors_in_offsets(offsets)
        lifted_offsets = offsets[np.logical_not(is_direct_neigh_offset)]

        add_lifted_edges = True
        if isinstance(self.offset_probabilities, np.ndarray):
            lifted_probs = self.offset_probabilities[np.logical_not(is_direct_neigh_offset)]
            # Check if we should add lifted edges at all:
            add_lifted_edges = any(lifted_probs != 0.)
            if add_lifted_edges:
                assert all(lifted_probs == 1.0), "Offset probabilities different from one are not supported" \
                                                 "when starting from a segmentation."


        lifted_graph, is_local_edge = build_lifted_graph_from_rag(
            rag,
            lifted_offsets,
            number_of_threads=self.n_threads,
            has_background_label=has_background_label,
            add_lifted_edges=add_lifted_edges
        )

        if self.verbose:
            print("Took {} s!".format(time.time() - tick))
            print("Computing edge_features...")
            tick = time.time()

        # Compute edge sizes and accumulate average:
        edge_indicators, edge_sizes = nrag.accumulate_affinities_mean_and_length(
            affinities,
            offsets,
            segmentation,
            graph=lifted_graph,
            offset_weights=offsets_weights,
            ignore_label=None, number_of_threads=self.n_threads
        )

        out_dict['graph'] = lifted_graph
        out_dict['edge_indicators'] = edge_indicators
        out_dict['edge_sizes'] = edge_sizes

        if not self.return_dict:
            edge_features = np.stack([edge_indicators, edge_sizes, is_local_edge])
            return lifted_graph, edge_features
        else:
            out_dict['is_local_edge'] = is_local_edge
            return out_dict


# def accumulate_affinities_on_graph_edges(affinities, offsets, label_image, graph=None,
#                                          mode="mean",
#                                          number_of_threads=6,
#                                          offsets_weights=None):
#     """
#     Label image or graph should be passed. Using nifty rag or undirected graph.
#
#     :param affinities: expected to have the offset dimension as last/first one
#     """
#     assert mode in ['mean'], "Only mean is implemented"
#
#     if affinities.shape[-1] != offsets.shape[0]:
#         assert affinities.shape[0] == offsets.shape[0], "Offsets do not match passed affs"
#         ndims = affinities.ndim
#         # Move first axis to the last dimension:
#         affinities = np.rollaxis(affinities, 0, ndims)
#
#     if graph is None:
#         graph = nrag.gridRag(label_image.astype(np.uint32))
#
#     if offsets_weights is not None:
#         if isinstance(offsets_weights, (list, tuple)):
#             offsets_weights = np.array(offsets_weights)
#         assert offsets_weights.shape[0] == affinities.shape[-1]
#         if all([w >= 1.0 for w in offsets_weights]):
#             # Take the inverse:
#             offsets_weights = 1. / offsets_weights
#         else:
#             assert all([w <= 1.0 for w in offsets_weights]) and all([w >= 0.0 for w in offsets_weights])
#     else:
#         offsets_weights = np.ones(affinities.shape[-1])
#
#     mean, count = nrag.accumulate_affinities_mean_and_length(random_affinities, offsets, random_labels,
#                                                              offset_weights=[2, 4, 1, 5])
#
#     accumulated_feat, counts, max_affinities = nrag.accumulateAffinitiesMeanAndLength(graph,
#                                                                                       label_image.astype(np.int32),
#                                                                                       affinities.astype(np.float32),
#                                                                                       offsets.astype(np.int32),
#                                                                                       offsets_weights.astype(
#                                                                                           np.float32),
#                                                                                       number_of_threads)
#     if mode == 'mean':
#         return accumulated_feat, counts
#     elif mode == 'max':
#         return accumulated_feat, max_affinities
