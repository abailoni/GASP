import numpy as np

from nifty import tools as ntools

from .core import run_GASP
from ...affinities.accumulator import AccumulatorLongRangeAffs
from ...affinities.utils import probs_to_costs
from ...utils.graph import build_pixel_lifted_graph_from_offsets
from ...utils.various import check_offsets

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
                 offsets_weights=None):
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
             the graph) should be weighted in the average-accumulation during the accumulation, related to the input
             `edge_sizes` of run_GASP(). By default all edges are weighted equally.
        """
        offsets = check_offsets(offsets)
        self.offsets = offsets

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
        if superpixel_generator is not None:
            if hasattr(superpixel_generator, "invert_affinities"):
                self.superpixel_generator.invert_affinities = False
            self.featurer = AccumulatorLongRangeAffs(self.offsets,
                                                     offsets_weights=self.offsets_weights,
                                                     used_offsets=self.used_offsets,
                                                     verbose=self.verbose,
                                                     n_threads=self.n_threads,
                                                     invert_affinities=False,
                                                     statistic='mean',
                                                     offset_probabilities=self.offsets_probabilities,
                                                     return_dict=True)

    def __call__(self, affinities, *args_superpixel_gen):
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
            superpixel_segmentation = self.superpixel_generator(affinities_, *args_superpixel_gen)
            return self.run_GASP_from_superpixels(affinities_, superpixel_segmentation)
        else:
            return self.run_GASP_from_pixels(affinities_)

    def run_GASP_from_pixels(self, affinities):
        offsets = self.offsets
        offset_probabilities = self.offsets_probabilities
        offsets_weights = self.offsets_weights
        if self.used_offsets is not None:
            assert len(self.used_offsets) < self.offsets.shape[0]
            offsets = self.offsets[self.used_offsets]
            affinities = affinities[self.used_offsets]
            offset_probabilities = self.offsets_probabilities[self.used_offsets]
            if isinstance(offsets_weights, (list, tuple)):
                offsets_weights = np.array(offsets_weights)
            offsets_weights = offsets_weights[self.used_offsets]

        image_shape = affinities.shape[1:]

        # Build graph:
        graph, is_local_edge, _, edge_sizes = \
            build_pixel_lifted_graph_from_offsets(
                image_shape,
                offsets,
                offsets_probabilities=offset_probabilities,
                offsets_weights=offsets_weights
            )

        edge_weights = graph.edgeValues(np.rollaxis(affinities, 0, 4))

        # Compute log costs:
        if self.use_logarithmic_weights:
            log_costs = probs_to_costs(1 - edge_weights, beta=self.beta_bias)
            signed_weights = log_costs
        else:
            signed_weights = edge_weights - self.beta_bias

        # Run GASP:
        nodeSeg, runtime = run_GASP(graph,
                                    signed_weights,
                                    edge_sizes=edge_sizes,
                                    is_mergeable_edge=is_local_edge,
                                    verbose=self.verbose,
                                    **self.run_GASP_kwargs)

        # TODO: map ignore label -1 to 0!
        segmentation = nodeSeg.reshape(image_shape)

        return segmentation, runtime

    def run_GASP_from_superpixels(self, affinities, superpixel_segmentation):
        # Compute graph and edge weights by accumulating over the affinities:
        featurer_outputs = self.featurer(affinities, superpixel_segmentation)
        graph = featurer_outputs['graph']
        edge_indicators = featurer_outputs['edge_indicators']
        edge_sizes = featurer_outputs['edge_sizes']
        is_local_edge = featurer_outputs['is_local_edge']

        # Optionally, use logarithmic weights and apply bias parameter
        if self.use_logarithmic_weights:
            log_costs = probs_to_costs(1 - edge_indicators, beta=self.beta_bias)
            signed_weights = log_costs
        else:
            signed_weights = edge_indicators - self.beta_bias

        # Run GASP:
        node_labels, runtime = \
            run_GASP(graph, signed_weights,
                     edge_sizes=edge_sizes,
                     is_mergeable_edge=is_local_edge,
                     verbose=self.verbose,
                     **self.run_GASP_kwargs)

        # Map node labels back to the original superpixel segmentation:
        final_segm = ntools.mapFeaturesToLabelArray(
            superpixel_segmentation,
            np.expand_dims(node_labels, axis=-1),
            nb_threads=self.n_threads,
            fill_value=-1.,
            ignore_label=-1,
        )[..., 0].astype(np.int64)

        # Increase by one, so ignore label becomes 0:
        final_segm += 1

        return final_segm, runtime
