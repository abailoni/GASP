import time

import numpy as np
from nifty.graph import rag as nrag

from ..utils.graph import build_lifted_graph_from_rag, get_rag
from ..utils.various import check_offsets

class AccumulatorLongRangeAffs(object):
    def __init__(self, offsets,
                 offsets_weights=None,
                 used_offsets=None,
                 verbose=True,
                 n_threads=1,
                 invert_affinities=False,
                 statistic='mean',
                 offset_probabilities=None,
                 return_dict=False,
                 mask_used_edges=None):

        offsets = check_offsets(offsets)

        self.used_offsets = used_offsets
        self.return_dict = return_dict
        self.offsets_weights = offsets_weights
        self.statistic = statistic

        assert isinstance(n_threads, int)

        self.offsets = offsets
        self.verbose = verbose
        self.n_threads = n_threads
        self.invert_affinities = invert_affinities
        self.offset_probabilities = offset_probabilities
        self.mask_used_edges = mask_used_edges

    def __call__(self, affinities, segmentation):
        tick = time.time()
        offsets = self.offsets
        offsets_weights = self.offsets_weights
        if self.used_offsets is not None:
            assert len(self.used_offsets) < self.offsets.shape[0]
            offsets = self.offsets[self.used_offsets]
            affinities = affinities[self.used_offsets]
            # FIXME: problem is weights are None (and how should I define them? On the full offsets or just the used ones?)
            if isinstance(offsets_weights, (list, tuple)):
                offsets_weights = np.array(offsets_weights)
            # elif offsets_weights is None:
            #     offsets_weights = np.ones(off)
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
            print("Building graph...")
        lifted_graph, is_local_edge = build_lifted_graph_from_rag(
            rag,
            segmentation,
            offsets,
            offset_probabilities=self.offset_probabilities,
            number_of_threads=self.n_threads,
            has_background_label=has_background_label,
            mask_used_edges=self.mask_used_edges
        )

        if self.verbose:
            print("Took {} s!".format(time.time() - tick))
            print("Computing edge_features...")
            tick = time.time()

        # Compute edge sizes and accumulate average/max:
        edge_indicators, edge_sizes = \
            accumulate_affinities_on_graph_edges(
                affinities, offsets,
                label_image=segmentation,
                graph=lifted_graph,
                mode=self.statistic,
                offsets_weights=offsets_weights,
                number_of_threads=self.n_threads)
        out_dict['graph'] = lifted_graph
        out_dict['edge_indicators'] = edge_indicators
        out_dict['edge_sizes'] = edge_sizes

        if not self.return_dict:
            edge_features = np.stack([edge_indicators, edge_sizes, is_local_edge])
            return lifted_graph, edge_features
        else:
            out_dict['is_local_edge'] = is_local_edge
            return out_dict


def accumulate_affinities_on_graph_edges(affinities, offsets, label_image, graph=None,
                                         mode="mean",
                                         number_of_threads=6,
                                         offsets_weights=None):
    """
    Label image or graph should be passed. Using nifty rag or undirected graph.

    :param affinities: expected to have the offset dimension as last/first one
    """
    assert mode in ['mean', 'max'], "Only max and mean are implemented"

    if affinities.shape[-1] != offsets.shape[0]:
        assert affinities.shape[0] == offsets.shape[0], "Offsets do not match passed affs"
        ndims = affinities.ndim
        # Move first axis to the last dimension:
        affinities = np.rollaxis(affinities, 0, ndims)

    if graph is None:
        graph = nrag.gridRag(label_image.astype(np.uint32))

    if offsets_weights is not None:
        if isinstance(offsets_weights, (list, tuple)):
            offsets_weights = np.array(offsets_weights)
        assert offsets_weights.shape[0] == affinities.shape[-1]
        if all([w >= 1.0 for w in offsets_weights]):
            # Take the inverse:
            offsets_weights = 1. / offsets_weights
        else:
            assert all([w <= 1.0 for w in offsets_weights]) and all([w >= 0.0 for w in offsets_weights])
    else:
        offsets_weights = np.ones(affinities.shape[-1])

    accumulated_feat, counts, max_affinities = nrag.accumulateAffinitiesMeanAndLength(graph,
                                                                                      label_image.astype(np.int32),
                                                                                      affinities.astype(np.float32),
                                                                                      offsets.astype(np.int32),
                                                                                      offsets_weights.astype(
                                                                                          np.float32),
                                                                                      number_of_threads)
    if mode == 'mean':
        return accumulated_feat, counts
    elif mode == 'max':
        return accumulated_feat, max_affinities
