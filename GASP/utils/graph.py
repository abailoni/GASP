import time
import nifty
import vigra
import numpy as np
from nifty import graph as ngraph
from nifty.graph import rag as nrag
import warnings
from .various import check_offsets, find_indices_direct_neighbors_in_offsets

from affogato.affinities import compute_affinities

from elf.segmentation.features import compute_grid_graph_affinity_features
import numpy.ma as ma

def get_rag(segmentation, nb_threads):
    """
    If the segmentation has values equal to -1, those are interpreted as background pixels.

    When this rag is build, the node IDs will be taken from segmentation and the background_node will have ID
    previous_max_label+1

    In `build_lifted_graph_from_rag`, the background node and all the edges connecting to it are ignored while creating
    the new (possibly lifted) undirected graph.
    """

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
    """
    If has_background_label is true, it assumes that it has label rag.numberOfNodes - 1 (See function `get_rag`)
    The background node and all the edges connecting to it are ignored when creating
    the new (possibly lifted) undirected graph.
    -------

    """
    # TODO: in order to support an edge_mask, getting the lifted edges is the easy part, but then I also need to accumulate
    #   affinities properly (and ignore those not in the mask)
    # TODO: add options `set_only_local_connections_as_mergeable` similarly to `build_pixel_long_range_grid_graph_from_offsets`

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
        # Note that this function could return the same lifted edge multiple times, so I need to add them to the graph
        # to see how many will be actually added
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


def edge_mask_from_offsets_prob(shape, offsets_probabilities, edge_mask=None):
    shape = tuple(shape) if not isinstance(shape, tuple) else shape

    offsets_probabilities = np.require(offsets_probabilities, dtype='float32')
    nb_offsets = offsets_probabilities.shape[0]
    edge_mask = np.ones((nb_offsets,)+ shape, dtype='bool') if edge_mask is None else edge_mask
    assert (offsets_probabilities.min() >= 0.0) and (offsets_probabilities.max() <= 1.0)

    # Randomly sample some edges to add to the graph:
    edge_mask = []
    for off_prob in offsets_probabilities:
        edge_mask.append(np.random.random(shape) <= off_prob)
    edge_mask = np.logical_and(np.stack(edge_mask, axis=-1), edge_mask)

    return edge_mask

def from_foreground_mask_to_edge_mask(foreground_mask, offsets, mask_used_edges=None):
    _, valid_edges = compute_affinities(foreground_mask.astype('uint64'), offsets.tolist(), True, 0)

    if mask_used_edges is not None:
        return np.logical_and(valid_edges, mask_used_edges)
    else:
        return valid_edges.astype('bool')

def build_pixel_long_range_grid_graph_from_offsets(image_shape,
                                                   offsets,
                                                   affinities,
                                                   offsets_probabilities=None,
                                                   mask_used_edges=None,
                                                   offset_weights=None,
                                                   set_only_direct_neigh_as_mergeable=True,
                                                   foreground_mask=None):
    """
    Parameters
    ----------
    offset_weights: Defines the size of each edge in the graph, depending on the associated offset.
    """
    # TODO: add support for foreground mask (masked nodes are removed from final undirected graph

    image_shape = tuple(image_shape) if not isinstance(image_shape, tuple) else image_shape
    offsets = check_offsets(offsets)

    if foreground_mask is not None:
        # Mask edges connected to background:
        mask_used_edges = from_foreground_mask_to_edge_mask(foreground_mask, offsets, mask_used_edges=mask_used_edges)

    # Create temporary grid graph:
    grid_graph = ngraph.undirectedGridGraph(image_shape)

    # Compute edge mask from offset probs:
    if offsets_probabilities is not None:
        if mask_used_edges is not None:
            warnings.warn("!!! Warning: both edge mask and offsets probabilities were used!!!")
        mask_used_edges = edge_mask_from_offsets_prob(image_shape, offsets_probabilities, mask_used_edges)

    uv_ids, edge_weights = compute_grid_graph_affinity_features(grid_graph, affinities,
                                         offsets=offsets, mask=mask_used_edges)

    nb_nodes = grid_graph.numberOfNodes
    projected_node_ids_to_pixels = grid_graph.projectNodeIdsToPixels()

    if foreground_mask is not None:
        # Mask background nodes and relabel node ids continuous before to create final graph:
        projected_node_ids_to_pixels += 1
        projected_node_ids_to_pixels[np.invert(foreground_mask)] = 0
        projected_node_ids_to_pixels, new_max_label, mapping = vigra.analysis.relabelConsecutive(projected_node_ids_to_pixels,
                                                                                  keep_zeros=True)
        # The following assumes that previously computed edges has alreadyu been masked
        uv_ids += 1
        vigra.analysis.applyMapping(uv_ids, mapping, out=uv_ids)
        nb_nodes = new_max_label+1

    # Create new undirected graph with all edges (including long-range ones):
    graph = ngraph.UndirectedGraph(nb_nodes)
    graph.insertEdges(uv_ids)

    # By default every edge is local/mergable:
    is_local_edge = np.ones(graph.numberOfEdges, dtype='bool')
    if set_only_direct_neigh_as_mergeable:
        # Get edge ids of local edges:
        # Warning: do not use grid_graph.projectEdgeIdsToPixels because edges ids could be inconsistent with those created
        # with compute_grid_graph_affinity_features assuming the given offsets!
        is_dir_neighbor, _ = find_indices_direct_neighbors_in_offsets(offsets)
        projected_local_edge_ids = grid_graph.projectEdgeIdsToPixelsWithOffsets(np.array(offsets))[is_dir_neighbor]
        is_local_edge = np.isin(np.arange(edge_weights.shape[0]),
                                projected_local_edge_ids[projected_local_edge_ids != -1].flatten(),
                                assume_unique=True)


    edge_sizes = np.ones(graph.numberOfEdges, dtype='float32')
    # TODO: use np.unique or similar on edge indices, but only if offset_weights are given (expensive)
    # Get local edges:
    # grid_graph.projectEdgeIdsToPixels()
    assert offset_weights is None, "Not implemented yet"

    return graph, projected_node_ids_to_pixels, edge_weights, is_local_edge, edge_sizes
