import numpy as np
import time

try:
    from affogato import segmentation as aff_segm
except ImportError:
    aff_segm = None

import nifty.graph.agglo as nifty_agglo

def run_GASP(
        graph,
        signed_edge_weights,
        linkage_criteria='mean',
        add_cannot_link_constraints= False,
        edge_sizes=None,
        is_mergeable_edge=None,
        use_efficient_implementations=True,
        verbose=False,
        linkage_criteria_kwargs=None,
        merge_constrained_edges_at_the_end=False,
        export_agglomeration_data=False,
        print_every=100000):
    """
    Run the Generalized Algorithm for Agglomerative Clustering on Signed Graphs (GASP).
    The C++ implementation is currently part of the nifty library (https://github.com/abailoni/nifty).

    Parameters
    ----------
    graph : nifty.graph
        Instance of a graph, e.g. nifty.graph.UndirectedGraph, nifty.graph.undirectedLongRangeGridGraph or
        nifty.graph.rag.gridRag

    signed_edge_weights : numpy.array(float) with shape (nb_graph_edges, )
        Attractive weights are positive; repulsive weights are negative.

    linkage_criteria : str (default 'mean')
        Specifies the linkage criteria / update rule used during agglomeration.
        List of available criteria:
            - 'mean', 'average', 'avg'
            - 'max', 'single_linkage'
            - 'min', 'complete_linkage'
            - 'mutex_watershed', 'abs_max'
            - 'sum'
            - 'quantile', 'rank' keeps statistics in a histogram, with parameters:
                    * q : float (default 0.5 equivalent to the median)
                    * numberOfBins: int (default: 40)
            - 'generalized_mean', 'gmean' with parameters:
                    * p : float (default: 1.0)
                    * https://en.wikipedia.org/wiki/Generalized_mean
            - 'smooth_max', 'smax' with parameters:
                    * p : float (default: 0.0)
                    * https://en.wikipedia.org/wiki/Smooth_maximum

    add_cannot_link_constraints : bool

    edge_sizes : numpy.array(float) with shape (nb_graph_edges, )
        Depending on the linkage criteria, they can be used during the agglomeration to weight differently
        the edges  (e.g. with sum or avg linkage criteria). Commonly used with regionAdjGraphs when edges
        represent boundaries of different length between segments / super-pixels. By default, all edges have
        the same weighting.

    is_mergeable_edge : numpy.array(bool) with shape (nb_graph_edges, )
        Specifies if an edge can be merged or not. Sometimes some edges represent direct-neighbor relations
        and others describe long-range connections. If a long-range connection /edge is assigned to
        `is_mergeable_edge = False`, then the two associated nodes are not merged until they become
        direct neighbors and they get connected in the image-plane.
        By default all edges are mergeable.

    use_efficient_implementations : bool (default: True)
        In the following special cases, alternative efficient implementations are used:
            - 'abs_max' criteria: Mutex Watershed (https://github.com/hci-unihd/mutex-watershed.git)
            - 'max' criteria without cannot-link constraints: maximum spanning tree

    verbose : bool (default: False)

    linkage_criteria_kwargs : dict
        Additional optional parameters passed to the chosen linkage criteria (see previous list)

    print_every : int (default: 100000)
        After how many agglomeration iteration to print in verbose mode

    Returns
    -------
    node_labels : numpy.array(uint) with shape (nb_graph_nodes, )
        Node labels representing the final clustering

    runtime : float
    """

    if use_efficient_implementations and (linkage_criteria in ['mutex_watershed', 'abs_max'] or
                                          (linkage_criteria == 'max' and not add_cannot_link_constraints)):
        assert not export_agglomeration_data, "Exporting extra agglomeration data is not possible when using " \
                                            "the efficient implementation."
        if is_mergeable_edge is not None:
            if not is_mergeable_edge.all():
                print("WARNING: Efficient implementations only works when all edges are mergeable. "
                      "In this mode, lifted and local edges will be treated equally, so there could be final clusters "
                      "consisting of multiple components 'disconnennted' in the image plane.")
        # assert is_mergeable_edge is None, "Efficient implementations only works when all edges are mergeable"
        nb_nodes = graph.numberOfNodes
        uv_ids = graph.uvIds()
        mutex_edges = signed_edge_weights < 0.

        tick = time.time()
        # These implementations use the convention where all edge weights are positive
        assert aff_segm is not None, "For the efficient implementation of GASP, affogato module is needed"
        if linkage_criteria in ['mutex_watershed', 'abs_max']:
            node_labels = aff_segm.compute_mws_clustering(nb_nodes,
                                             uv_ids[np.logical_not(mutex_edges)],
                                             uv_ids[mutex_edges],
                                             signed_edge_weights[np.logical_not(mutex_edges)],
                                             -signed_edge_weights[mutex_edges])
        else:
            node_labels = aff_segm.compute_single_linkage_clustering(nb_nodes,
                                                        uv_ids[np.logical_not(mutex_edges)],
                                                        uv_ids[mutex_edges],
                                                        signed_edge_weights[np.logical_not(mutex_edges)],
                                                        -signed_edge_weights[mutex_edges])
        runtime = time.time() - tick
    else:
        cluster_policy = nifty_agglo.get_GASP_policy(graph, signed_edge_weights,
                                                     edge_sizes=edge_sizes,
                                                     linkage_criteria=linkage_criteria,
                                                     linkage_criteria_kwargs=linkage_criteria_kwargs,
                                                     add_cannot_link_constraints=add_cannot_link_constraints,
                                                     is_mergeable_edge=is_mergeable_edge,
                                                     merge_constrained_edges_at_the_end=merge_constrained_edges_at_the_end,
                                                     collect_stats_for_exported_data=export_agglomeration_data)
        agglomerativeClustering = nifty_agglo.agglomerativeClustering(cluster_policy)

        # Run clustering:
        tick = time.time()
        agglomerativeClustering.run(verbose=verbose,
                                    printNth=print_every)
        runtime = time.time() - tick

        # Collect results:
        node_labels = agglomerativeClustering.result()

        if export_agglomeration_data:
            exported_data = cluster_policy.exportAgglomerationData()

    if export_agglomeration_data:
        out_dict = {'agglomeration_data': exported_data}
        return node_labels, runtime, out_dict
    else:
        return node_labels, runtime
