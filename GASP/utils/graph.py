import time
import nifty
import numpy as np
from nifty import graph as ngraph
from nifty.graph import rag as nrag
import warnings
from .various import check_offsets, find_indices_direct_neighbors_in_offsets


def get_rag(segmentation, nb_threads):
    # Check if the segmentation has a background label that should be ignored in the graph:
    min_label = segmentation.min()
    if min_label >= 0:
        out_dict = {'has_background_label': False}
        return nrag.gridRag(segmentation.astype(np.uint32), numberOfThreads=nb_threads), out_dict
    else:
        assert min_label == -1, "The only accepted background label is -1"
        max_valid_label = segmentation.max()
        assert max_valid_label >= 0, "A label image with only background label was passed!"
        mod_segmentation = segmentation.copy()
        background_mask = segmentation == min_label
        mod_segmentation[background_mask] = max_valid_label + 1

        # Build rag including background:
        out_dict = {'has_background_label': True,
                    'updated_segmentation': mod_segmentation,
                    'background_label': max_valid_label + 1}
        return nrag.gridRag(mod_segmentation.astype(np.uint32), numberOfThreads=nb_threads), out_dict


def build_lifted_graph_from_rag(rag,
                                offsets,
                                number_of_threads=-1,
                                has_background_label=False,
                                add_lifted_edges=True):
    if not has_background_label:
        nb_local_edges = rag.numberOfEdges
        final_graph = rag
    else:
        # Find edges not connected to the background:
        edges = rag.uvIds()
        background_label = rag.numberOfNodes - 1
        valid_edges = edges[np.logical_and(edges[:, 0] != background_label, edges[:, 1] != background_label)]

        # Construct new graph without the background:
        new_graph = nifty.graph.undirectedGraph(rag.numberOfNodes - 1)
        new_graph.insertEdges(valid_edges)

        nb_local_edges = valid_edges.shape[0]
        final_graph = new_graph

    if not add_lifted_edges:
        return final_graph, np.ones((nb_local_edges,), dtype='bool')
    else:
        if not has_background_label:
            local_edges = rag.uvIds()
            final_graph = nifty.graph.undirectedGraph(rag.numberOfNodes)
            final_graph.insertEdges(local_edges)

        # Find lifted edges:
        possibly_lifted_edges = ngraph.rag.compute_lifted_edges_from_rag_and_offsets(rag,
                                                                                     offsets,
                                                                                     numberOfThreads=number_of_threads)

        # Delete lifted edges connected to the background label:
        if has_background_label:
            possibly_lifted_edges = possibly_lifted_edges[
                np.logical_and(possibly_lifted_edges[:, 0] != background_label,
                               possibly_lifted_edges[:, 1] != background_label)]

        final_graph.insertEdges(possibly_lifted_edges)
        total_nb_edges = final_graph.numberOfEdges

        is_local_edge = np.zeros(total_nb_edges, dtype=np.int8)
        is_local_edge[:nb_local_edges] = 1

        return final_graph, is_local_edge


def build_pixel_long_range_grid_graph_from_offsets(image_shape,
                                                   offsets,
                                                   offsets_probabilities=None,
                                                   mask_used_edges=None,
                                                   offset_weights=None,
                                                   set_only_direct_neigh_as_mergeable=True):
    """
    Parameters
    ----------
    offset_weights: Defines the size of each edge in the graph, depending on the associated offset.


    """
    image_shape = tuple(image_shape) if not isinstance(image_shape, tuple) else image_shape
    offsets = check_offsets(offsets)

    graph = ngraph.undirectedLongRangeGridGraph(image_shape, offsets,
                                                offsets_probabilities=offsets_probabilities,
                                                edge_mask=mask_used_edges)

    # By default every edge is local/mergable:
    is_local_edge = np.ones(graph.numberOfEdges, dtype='bool')

    edge_offset_index = graph.edgeOffsetIndex()

    if set_only_direct_neigh_as_mergeable:
        # Assume that the number of local offsets are equal to the dimension of the image:
        is_local_edge[:] = False
        _, indices_local_offsets = find_indices_direct_neighbors_in_offsets(offsets)
        for local_offset in indices_local_offsets:
            is_local_edge[edge_offset_index == local_offset] = True

    edge_sizes = np.ones(graph.numberOfEdges, dtype='float32')
    if offset_weights is not None:
        offset_weights = np.require(offset_weights, dtype='float32')
        edge_sizes = offset_weights[edge_offset_index]

    return graph, is_local_edge, edge_sizes
