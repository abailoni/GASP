import numpy as np
import warnings
import time

from nifty import tools as ntools
from affogato.affinities import compute_affinities
try:
    from affogato.segmentation import compute_mws_segmentation_from_affinities
except ImportError:
    compute_mws_segmentation_from_affinities = None
from .core import run_GASP
from ...affinities.accumulator import AccumulatorLongRangeAffs
from ...affinities.utils import probs_to_costs
from ...utils.graph import build_pixel_long_range_grid_graph_from_offsets
from ...utils.various import check_offsets, find_indices_direct_neighbors_in_offsets


class GaspFromAffinities(object):
    def __init__(self,
                 offsets,
                 beta_bias=0.5,
                 superpixel_generator=None,
                 run_GASP_kwargs=None,
                 n_threads=1,
                 verbose=False,
                 invert_affinities=False,
                 offsets_probabilities=None,
                 use_logarithmic_weights=False,
                 used_offsets=None,
                 offsets_weights=None,
                 return_extra_outputs=False,
                 set_only_direct_neigh_as_mergeable=True,
                 ignore_edge_sizes=False, # TODO: delete. It should never be True for superpixels...
                 ):
        """
        Run the Generalized Algorithm for Signed Graph Agglomerative Partitioning from affinities computed from
        an image. The clustering can be both initialized from pixels and superpixels.

        Parameters
        ----------
        offsets :  np.array(int) or list
            Array with shape (nb_offsets, nb_dimensions). Example with three direct neighbors in 3D:
                [ [-1, 0, 0],
                  [0, -1, 0],
                  [0, 0, -1]  ]

        beta_bias : float (default: 0.5)
            Add bias to the edge weights

        superpixel_generator : callable (default: None)
            Callable with inputs (affinities, *args_superpixel_gen). If None, run_GASP() is initialized from pixels.

        run_GASP_kwargs : dict (default: None)
            Additional arguments to be passed to run_GASP()

        n_threads :  int (default: 1)

        verbose : bool (default: False)

        invert_affinities : bool (default: False)

        offsets_probabilities : np.array(float) or list
            Array with shape (nb_offsets), specifying the probabilities with which each type of edge-connection
            should be added to the graph. BY default all connections are added.

        used_offsets : np.array(int) or list
            Array with shape (nb_offsets), specifying which offsets (i.e. which channels in the affinities array)
            should be considered to accumulate the average over the initial superpixel-boundaries.
            By default all offsets are used.

        offsets_weights : np.array(float) or list
            Array with shape (nb_offsets), specifying how each offset (i.e. a type of edge-connection in
             the graph) should be weighted in the average-accumulation during the accumulation,
             related to the input `edge_sizes` of run_GASP(). By default all edges are weighted equally.
        """
        offsets = check_offsets(offsets)
        self.offsets = offsets

        # Parse inputs:
        if offsets_probabilities is not None:
            if isinstance(offsets_probabilities, (float, int)):
                is_offset_direct_neigh, _ = find_indices_direct_neighbors_in_offsets(offsets)
                offsets_probabilities = np.ones((offsets.shape[0],), dtype="float32") * offsets_probabilities
                # Direct neighbors should be always added:
                offsets_probabilities[is_offset_direct_neigh] = 1.
            else:
                offsets_probabilities = np.require(offsets_probabilities, dtype='float32')
                assert len(offsets_probabilities) == len(offsets)

        if offsets_weights is not None:
            offsets_weights = np.require(offsets_weights, dtype='float32')
            assert len(offsets_weights) == len(offsets)

        if used_offsets is not None:
            assert len(used_offsets) < offsets.shape[0]
            if offsets_probabilities is not None:
                offsets_probabilities = offsets_probabilities[used_offsets]
            if offsets_weights is not None:
                offsets_weights = offsets_weights[used_offsets]

        self.offsets_probabilities = offsets_probabilities
        self.used_offsets = used_offsets
        self.offsets_weights = offsets_weights


        assert isinstance(n_threads, int)
        self.n_threads = n_threads

        assert isinstance(invert_affinities, bool)
        self.invert_affinities = invert_affinities

        assert isinstance(verbose, bool)
        self.verbose = verbose

        run_GASP_kwargs = run_GASP_kwargs if isinstance(run_GASP_kwargs, dict) else {}
        self.run_GASP_kwargs = run_GASP_kwargs

        assert (beta_bias <= 1.0) and (
                beta_bias >= 0.), "The beta bias parameter is expected to be in the interval (0,1)"
        self.beta_bias = beta_bias

        assert isinstance(use_logarithmic_weights, bool)
        self.use_logarithmic_weights = use_logarithmic_weights

        self.superpixel_generator = superpixel_generator
        self.return_extra_outputs = return_extra_outputs
        self.set_only_direct_neigh_as_mergeable = set_only_direct_neigh_as_mergeable
        self.ignore_edge_sizes = ignore_edge_sizes

    def __call__(self, affinities, *args_superpixel_gen,
                 mask_used_edges=None, affinities_weights=None, foreground_mask=None):
        """
        Parameters
        ----------
        affinities : np.array(float)
            Array with shape (nb_offsets, ) + shape_image, where the shape of the image can be 2D or 3D.
            Passed values should be in interval [0, 1], where 1-values should represent intra-cluster connections
            (high affinity, merge) and 0-values inter-cluster connections (low affinity, boundary evidence, split).

        args_superpixel_gen :
            Additional arguments passed to the superpixel generator

        Returns
        -------
        final_segmentation : np.array(int)
            Array with shape shape_image.

        runtime : float
        """
        assert isinstance(affinities, np.ndarray)
        assert affinities.ndim == 4, "Need affinities with 4 channels, got %i" % affinities.ndim
        if self.invert_affinities:
            affinities_ = 1. - affinities
        else:
            affinities_ = affinities

        if self.superpixel_generator is not None:
            superpixel_segmentation = self.superpixel_generator(affinities_, *args_superpixel_gen, foreground_mask=foreground_mask)
            return self.run_GASP_from_superpixels(affinities_, superpixel_segmentation,
                                                  mask_used_edges=mask_used_edges,
                                                  affinities_weights=affinities_weights, foreground_mask=foreground_mask)
        else:
            return self.run_GASP_from_pixels(affinities_, mask_used_edges=mask_used_edges,
                                             affinities_weights=affinities_weights, foreground_mask=foreground_mask)

    def run_GASP_from_pixels(self, affinities, mask_used_edges=None, foreground_mask=None,
                             affinities_weights=None):
        assert affinities_weights is None, "Not yet implemented from pixels"
        assert affinities.shape[0] == len(self.offsets)
        offsets = self.offsets
        if self.used_offsets is not None:
            affinities = affinities[self.used_offsets]
            offsets = offsets[self.used_offsets]
            if mask_used_edges is not None:
                mask_used_edges = mask_used_edges[self.used_offsets]

        if foreground_mask is not None:
            mask_used_edges = self.from_foreground_mask_to_edge_mask(foreground_mask, offsets, mask_used_edges=mask_used_edges)

        image_shape = affinities.shape[1:]

        # Check if I should use efficient implementation of the MWS:
        run_kwargs = self.run_GASP_kwargs
        export_agglomeration_data = run_kwargs.get("export_agglomeration_data", False)
        if run_kwargs.get("use_efficient_implementations", True) and run_kwargs.get("linkage_criteria") in ['mutex_watershed', 'abs_max']:
            assert compute_mws_segmentation_from_affinities is not None, "Efficient MWS implementation not available. Update the affogato repository "
            assert not export_agglomeration_data, "Exporting extra agglomeration data is not possible when using " \
                                                  "the efficient implementation of MWS."
            if self.set_only_direct_neigh_as_mergeable:
                warnings.warn("With efficient implementation of MWS, it is not possible to set only direct neighbors"
                              "as mergeable.")
            tick = time.time()
            segmentation, valid_edge_mask = compute_mws_segmentation_from_affinities(affinities, offsets,
                                                     beta_parameter=self.beta_bias,
                                                     foreground_mask=foreground_mask, edge_mask=mask_used_edges,
                                                                    return_valid_edge_mask=True)
            runtime = time.time() - tick
            if self.return_extra_outputs:
                MC_energy = self.get_multicut_energy_segmentation(segmentation,affinities,
                                                                  offsets, valid_edge_mask)
                out_dict = {'runtime': runtime,
                            'multicut_energy': MC_energy}
                return segmentation, out_dict
            else:
                return segmentation, runtime

        # Build graph:
        if self.verbose:
            print("Building graph...")
        graph, is_local_edge, edge_sizes = \
            build_pixel_long_range_grid_graph_from_offsets(
                image_shape,
                offsets,
                offsets_probabilities=self.offsets_probabilities,
                mask_used_edges=mask_used_edges,
                offset_weights=self.offsets_weights,
                set_only_direct_neigh_as_mergeable=self.set_only_direct_neigh_as_mergeable
            )

        edge_weights = graph.edgeValues(np.rollaxis(affinities, 0, 4))
        # counts = np.unique(edge_weights, return_counts=True)[1]
        # print("Max count: ", counts.max())

        # Compute log costs:
        log_costs = probs_to_costs(1 - edge_weights, beta=self.beta_bias)
        if self.use_logarithmic_weights:
            signed_weights = log_costs
        else:
            signed_weights = edge_weights - self.beta_bias

        if self.ignore_edge_sizes:
            edge_sizes = np.ones_like(edge_sizes)

        # Run GASP:
        if self.verbose:
            print("Start agglo...")

        outputs = run_GASP(graph,
                                    signed_weights,
                                    edge_sizes=edge_sizes,
                                    is_mergeable_edge=is_local_edge,
                                    verbose=self.verbose,
                                    **self.run_GASP_kwargs)

        if export_agglomeration_data:
            nodeSeg, runtime, exported_data = outputs
        else:
            exported_data = {}
            nodeSeg, runtime = outputs

        # TODO: apply foreground_mask again
        segmentation = nodeSeg.reshape(image_shape)

        if foreground_mask is not None:
            zero_mask = segmentation == 0
            segmentation[foreground_mask == 0] = 0
            if np.any(zero_mask):
                max_label = segmentation.max()
                warnings.warn("Zero segment remapped to {} in final segmentation".format(max_label + 1))
                segmentation[zero_mask] = max_label + 1


        if self.return_extra_outputs:
            MC_energy = self.get_multicut_energy(graph, nodeSeg, log_costs, edge_sizes)
            out_dict = {"multicut_energy": MC_energy,
                        "runtime": runtime,
                        "graph": graph,
                        "is_local_edge": is_local_edge,
                        "edge_sizes": edge_sizes
                        }
            if export_agglomeration_data:
                out_dict.update(exported_data)
            return segmentation, out_dict
        else:
            if export_agglomeration_data:
                warnings.warn("In order to export agglomeration data, also set the `return_extra_outputs` to True")
            return nodeSeg, runtime

    def run_GASP_from_superpixels(self, affinities, superpixel_segmentation, foreground_mask=None,
                                  mask_used_edges=None, affinities_weights=None):
        # TODO: compute affiniteis_weights automatically from segmentation if needed
        # When I will implement the mask_edge, remeber to crop it depending on the used offsets
        assert mask_used_edges is None, "Edge mask cannot be used when starting from a segmentation."
        assert self.set_only_direct_neigh_as_mergeable, "Not implemented atm from superpixels"
        featurer = AccumulatorLongRangeAffs(self.offsets,
                                            offsets_weights=self.offsets_weights,
                                            used_offsets=self.used_offsets,
                                            verbose=self.verbose,
                                            n_threads=self.n_threads,
                                            invert_affinities=False,
                                            statistic='mean',
                                            offset_probabilities=self.offsets_probabilities,
                                            return_dict=True)

        # Compute graph and edge weights by accumulating over the affinities:
        featurer_outputs = featurer(affinities, superpixel_segmentation,
                                    affinities_weights=affinities_weights)
        graph = featurer_outputs['graph']
        edge_indicators = featurer_outputs['edge_indicators']
        edge_sizes = featurer_outputs['edge_sizes']
        is_local_edge = featurer_outputs['is_local_edge']

        if self.ignore_edge_sizes:
            edge_sizes = np.ones_like(edge_sizes)

        # Optionally, use logarithmic weights and apply bias parameter
        log_costs = probs_to_costs(1 - edge_indicators, beta=self.beta_bias)
        if self.use_logarithmic_weights:
            signed_weights = log_costs
        else:
            signed_weights = edge_indicators - self.beta_bias

        # Run GASP:
        export_agglomeration_data = self.run_GASP_kwargs.get("export_agglomeration_data", False)
        outputs = \
            run_GASP(graph, signed_weights,
                     edge_sizes=edge_sizes,
                     is_mergeable_edge=is_local_edge,
                     verbose=self.verbose,
                     **self.run_GASP_kwargs)

        if export_agglomeration_data:
            node_labels, runtime, exported_data = outputs
        else:
            exported_data = {}
            node_labels, runtime = outputs


        # Map node labels back to the original superpixel segmentation:
        final_segm = ntools.mapFeaturesToLabelArray(
            superpixel_segmentation,
            np.expand_dims(node_labels, axis=-1),
            nb_threads=self.n_threads,
            fill_value=-1.,
            ignore_label=-1,
        )[..., 0].astype(np.int64)

        # If there was a background label, reset it to zero:
        min_label = final_segm.min()
        if min_label < 0:
            assert min_label == -1

            # Move bacground label to 0, and map 0 segment (if any) to MAX_LABEL+1.
            # In this way, most of the final labels will stay consistent with the graph and given over-segmentation
            background_mask = final_segm == min_label
            zero_mask = final_segm == 0
            if np.any(zero_mask):
                max_label = final_segm.max()
                warnings.warn("Zero segment remapped to {} in final segmentation".format(max_label+1))
                final_segm[zero_mask] = max_label + 1
            final_segm[background_mask] = 0


        if self.return_extra_outputs:
            MC_energy = self.get_multicut_energy(graph, node_labels, log_costs, edge_sizes)
            out_dict = {"multicut_energy": MC_energy,
                        "runtime": runtime,
                        "graph": graph,
                        "is_local_edge": is_local_edge,
                        "edge_sizes": edge_sizes
                        }
            if export_agglomeration_data:
                out_dict.update(exported_data)
            return final_segm, out_dict
        else:
            if export_agglomeration_data:
                warnings.warn("In order to export agglomeration data, also set the `return_extra_outputs` to True")
            return final_segm, runtime


    def get_multicut_energy(self, graph, node_segm, log_edge_weights, edge_sizes=None):
        if edge_sizes is None:
            edge_sizes = np.ones_like(log_edge_weights)
        edge_labels = graph.nodeLabelsToEdgeLabels(node_segm)
        return (log_edge_weights * edge_labels * edge_sizes).sum()

    def get_multicut_energy_segmentation(self, pixel_segm, affinities, offsets, edge_mask=None):
        if edge_mask is None:
            edge_mask = np.ones_like(affinities, dtype='bool')

        log_affinities = probs_to_costs(1 - affinities, beta=self.beta_bias)

        # Find affinities "on cut":
        affs_not_on_cut, _ = compute_affinities(pixel_segm.astype('uint64'), offsets.tolist(), False, 0)
        return log_affinities[np.logical_and(affs_not_on_cut == 0, edge_mask)].sum()

    def from_foreground_mask_to_edge_mask(self, foreground_mask, offsets, mask_used_edges=None):
        _, valid_edges = compute_affinities(foreground_mask.astype('uint64'), offsets.tolist(), True, 0)

        if mask_used_edges is not None:
            return np.logical_and(valid_edges, mask_used_edges)
        else:
            return valid_edges.astype('bool')

        pass


class SegmentationFeeder(object):
    """
    A simple function that expects affinities and initial segmentation (with optional foreground mask)
    and can be used as "superpixel_generator" for GASP
    """
    def __call__(self, affinities, segmentation, foreground_mask=None):
        if foreground_mask is not None:
            assert foreground_mask.shape == segmentation.shape
            segmentation = segmentation.astype('int64')
            segmentation = np.where(foreground_mask, segmentation, np.ones_like(segmentation) * (-1))
        return segmentation
