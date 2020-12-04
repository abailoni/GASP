"""Uses watershed implementation in vigra library"""
import numpy as np
import vigra
from vigra.analysis import watershedsNew
from scipy.ndimage.filters import median_filter

from nifty import tools as ntools

from ..affinities.utils import superpixel_stacked, from_affinities_to_hmap



import nifty.graph.rag as nrag


class WatershedFromAffinities(object):
    def __init__(self, offsets,
                       used_offsets=None,
                       offset_weights=None,
                       stacked_2d=True,
                       invert_affinities=False,
                       return_hmap=False,
                       n_threads=1):
        """
        :param invert_affinities: by default it uses affinities (1: merge, 0: split).
                Set to True if necessary.
        :param offsets: np.array or list
        :param used_offsets: list of offset indices
        :param offset_weights: list of weights
        """
        if isinstance(offsets, list):
            offsets = np.array(offsets)
        else:
            assert isinstance(offsets, np.ndarray)

        self.offsets = offsets
        # Consistency of these inputs is checked in from_affinities_to_hmap
        self.used_offsets = used_offsets
        self.offset_weights = offset_weights

        self.invert_affinities = invert_affinities
        self.stacked_2d = stacked_2d
        self.n_threads = n_threads
        self.return_hmap = return_hmap



    def ws_superpixels(self, hmap_z_slice):
        assert hmap_z_slice.ndim ==  2 or hmap_z_slice.ndim == 3

        # hmap_z_slice = median_filter(hmap_z_slice, 1)

        segmentation, max_label = watershedsNew(hmap_z_slice)
        return segmentation, max_label

    def __call__(self, affinities, *args):
        """
        Here we expect real affinities (1: merge, 0: split).
        If the opposite is passed, set option `invert_affinities == True`
        """
        foreground_mask = None
        # TODO: update with only one optional arg...
        if len(args) != 0:
            assert len(args) == 1
            foreground_mask = args[0]


        assert affinities.shape[0] == len(self.offsets)
        assert affinities.ndim == 4

        if self.invert_affinities:
            affinities = 1. - affinities

        hmap = from_affinities_to_hmap(affinities, self.offsets, self.used_offsets,
                                self.offset_weights)

        if self.stacked_2d:
            segmentation, _ = superpixel_stacked(hmap, self.ws_superpixels, self.n_threads)
        else:
            segmentation, _ = self.ws_superpixels(hmap)

        # Mask with background (e.g. ignore GT-label):
        if foreground_mask is not None:
            assert foreground_mask.shape == segmentation.shape
            segmentation = segmentation.astype('int64')
            segmentation = np.where(foreground_mask, segmentation, np.ones_like(segmentation) * (-1))

        if self.return_hmap:
            return segmentation, hmap
        else:
            return segmentation


class SizeThreshAndGrowWithWS(object):
    """
    Ignore all segments smaller than a certain size threshold and
    then grow remaining segments with seeded WS.

    Segments are grown on every slice in 2D.
    """
    def __init__(self, size_threshold,
                 offsets,
                 hmap_kwargs=None,
                 apply_WS_growing=True,
                 size_of_2d_slices=False,
                 debug=False,
                 with_background=False):
        """
        :param apply_WS_growing: if False, then the 'seed_mask' is returned
        :param size_of_2d_slices: compute size for all z-slices (memory efficient)
        """
        self.size_threshold = size_threshold
        self.offsets = offsets
        assert len(offsets[0]) ==  3, "Only 3D supported atm"
        self.hmap_kwargs = {} if hmap_kwargs is None else hmap_kwargs
        self.apply_WS_growing = apply_WS_growing
        self.debug = debug
        self.size_of_2d_slices = size_of_2d_slices
        self.with_background = with_background

    def __call__(self, affinities, label_image):
        assert len(self.offsets) == affinities.shape[0], "Affinities does not match offsets"
        if self.debug:
            print("Computing segment sizes...")
        label_image = label_image.astype(np.uint32)

        def get_size_map(label_image):
            node_sizes = np.bincount(label_image.flatten())
            # rag = nrag.gridRag(label_image)
            # _, node_features = nrag.accumulateMeanAndLength(rag, label_image.astype('float32'),
            #                                                 blockShape=[1, 100, 100],
            #                                                 numberOfThreads=8,
            #                                                 saveMemory=True)
            # nodeSizes = node_features[:, [1]]
            return ntools.mapFeaturesToLabelArray(label_image, node_sizes[:,None], nb_threads=6).squeeze()



        if not self.size_of_2d_slices:
            sizeMap = get_size_map(label_image)
        else:
            sizeMap = np.empty_like(label_image)
            for z in range(label_image.shape[0]):
                sizeMap[[z]] = get_size_map(label_image[[z]])
                print(z, flush=True, end=" ")


        sizeMask = sizeMap > self.size_threshold
        seeds = ((label_image+1)*sizeMask).astype(np.uint32)

        background_mask = None
        if self.with_background:
            background_mask = label_image == 0
            seeds[background_mask] = 0

        if not self.apply_WS_growing:
            return seeds
        else:
            if self.debug:
                print("Computing hmap and WS...")
            hmap = from_affinities_to_hmap(affinities, self.offsets, **self.hmap_kwargs)
            watershedResult = np.empty_like(seeds)
            for z in range(hmap.shape[0]):
                watershedResult[z], _ = vigra.analysis.watershedsNew(hmap[z], seeds=seeds[z],
                                                                     method='RegionGrowing')
                if self.with_background:
                    watershedResult[z][background_mask[z]] = 0

            # Re-normalize indices numbers:
            if self.with_background:
                return vigra.analysis.labelVolumeWithBackground(watershedResult.astype(np.uint32))
            else:
                return vigra.analysis.labelVolume(watershedResult.astype(np.uint32))



class SeededWatershedOnAffinities(object):
    """
    Grow segments in given segmentation to fill the background in the image
    (wrapper of vigra seeded watershed).

    Segments are grown on every slice in 2D. Background label is assumed to be zero.
    """
    def __init__(self,
                 offsets,
                 hmap_kwargs=None,
                 debug=False):
        self.offsets = offsets
        assert len(offsets[0]) ==  3, "Only 3D supported atm"
        self.hmap_kwargs = {} if hmap_kwargs is None else hmap_kwargs
        self.debug = debug

    def __call__(self, affinities, label_image):
        assert len(self.offsets) == affinities.shape[0], "Affinities does not match offsets"

        assert label_image.min() >= 0, "Negative labels passed"

        label_image = label_image.astype(np.uint32)

        foreground_mask = label_image != 0
        seeds = label_image

        if self.debug:
            print("Computing hmap and WS...")

        hmap = from_affinities_to_hmap(affinities, self.offsets, **self.hmap_kwargs)
        watershedResult = np.empty_like(seeds)
        for z in range(hmap.shape[0]):
            watershedResult[z], _ = vigra.analysis.watershedsNew(hmap[z], seeds=seeds[z],
                                                                 method='RegionGrowing')

        return vigra.analysis.labelVolume(watershedResult.astype(np.uint32))
