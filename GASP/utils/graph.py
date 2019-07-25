import time
import nifty
import numpy as np
from nifty import graph as ngraph
from nifty.graph import rag as nrag


def get_rag(segmentation, nb_threads):
    # Check if the segmentation has a background label that should be ignored in the graph:
    min_label = segmentation.min()
    if min_label >= 0:
        return nrag.gridRag(segmentation.astype(np.uint32), numberOfThreads=nb_threads), False
    else:
        # print("RAG built with background label 0!")
        assert min_label == -1, "The only accepted background label is -1"
        max_valid_label = segmentation.max()
        assert max_valid_label >= 0, "A label image with only background label was passed!"
        mod_segmentation = segmentation.copy()
        background_mask = segmentation == min_label
        mod_segmentation[background_mask] = max_valid_label + 1

        # Build rag including background:
        return nrag.gridRag(mod_segmentation.astype(np.uint32), numberOfThreads=nb_threads), True


def build_lifted_graph_from_rag(rag,
                                label_image,
                                offsets, offset_probabilities=None,
                                number_of_threads=8,
                                has_background_label=False,
                                mask_used_edges=None):
    if isinstance(offset_probabilities, np.ndarray):
        only_local = all(offset_probabilities == 0.)
    else:
        only_local = offset_probabilities == 0.

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

    if only_local:
        return final_graph, np.ones((nb_local_edges,), dtype='bool')
    else:
        if not has_background_label:
            local_edges = rag.uvIds()
            final_graph = nifty.graph.undirectedGraph(rag.numberOfNodes)
            final_graph.insertEdges(local_edges)

        # Find lifted edges:
        used_offsets = offsets
        possibly_lifted_edges = ngraph.compute_lifted_edges_from_rag_and_offsets(rag,
                                                                                 label_image,
                                                                                 used_offsets,
                                                                                 offsets_probabilities=offset_probabilities,
                                                                                 number_of_threads=number_of_threads,
                                                                                 mask_used_edges=mask_used_edges)

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


def build_pixel_lifted_graph_from_offsets(image_shape,
                                          offsets,
                                          label_image=None,
                                          GT_label_image=None,
                                          offsets_probabilities=None,
                                          offsets_weights=None,
                                          strides=None,
                                          nb_local_offsets=3,
                                          downscaling_factor=None,
                                          mask_used_edges=None):
    if downscaling_factor is not None:
        raise NotImplementedError()

    image_shape = tuple(image_shape) if not isinstance(image_shape, tuple) else image_shape

    is_local_offset = np.zeros(offsets.shape[0], dtype='bool')
    is_local_offset[:nb_local_offsets] = True
    if label_image is not None:
        assert image_shape == label_image.shape
        if offsets_weights is not None:
            print("Offset weights ignored...!")

    graph = ngraph.undirectedLongRangeGridGraph(image_shape, offsets, is_local_offset,
                                                offsets_probabilities=offsets_probabilities,
                                                labels=label_image,
                                                strides=strides,
                                                mask_used_edges=mask_used_edges)
    nb_nodes = graph.numberOfNodes

    if label_image is None:
        offset_index = graph.edgeOffsetIndex()
        is_local_edge = np.empty_like(offset_index, dtype='bool')
        w = np.where(offset_index < nb_local_offsets)
        is_local_edge[:] = 0
        is_local_edge[w] = 1
    else:
        is_local_edge = graph.findLocalEdges(label_image).astype('int32')

    if offsets_weights is None or label_image is not None:
        edge_sizes = np.ones(graph.numberOfEdges, dtype='int32')
    else:
        if isinstance(offsets_weights, (list, tuple)):
            offsets_weights = np.array(offsets_weights)
        assert offsets_weights.shape[0] == offsets.shape[0]

        if offsets_weights.ndim == len(image_shape) + 1:
            edge_sizes = graph.edgeValues(np.rollaxis(offsets_weights, 0, 4))
        else:
            if all([w >= 1.0 for w in offsets_weights]):
                # Take the inverse:
                offsets_weights = 1. / offsets_weights
            else:
                assert all([w <= 1.0 for w in offsets_weights]) and all([w >= 0.0 for w in offsets_weights])

            edge_sizes = offsets_weights[offset_index.astype('int32')]

    if GT_label_image is None:
        GT_labels_nodes = np.zeros(nb_nodes, dtype=np.int64)
    else:
        assert GT_label_image.shape == image_shape
        GT_labels_image = GT_label_image.astype(np.uint64)
        GT_labels_nodes = graph.nodeValues(GT_labels_image)

    return graph, is_local_edge, GT_labels_nodes, edge_sizes
