import numpy as np
import vigra

from ...features import from_affinities_to_hmap


class IntersectWithBoundaryPixels(object):
    def __init__(self, offsets,
                 boundary_threshold=0.5, # 1.0 all boundary, 0.0 no boundary
                 used_offsets=None,
                 offset_weights=None):
        self.offsets = offsets
        self.used_offsets = used_offsets
        self.offset_weights = offset_weights
        self.boundary_threshold = boundary_threshold

    def __call__(self, affinities, segmentation):
        hmap = from_affinities_to_hmap(affinities, self.offsets, self.used_offsets,
                                       self.offset_weights)
        pixel_segm = np.arange(np.prod(segmentation.shape), dtype='uint64').reshape(segmentation.shape) + segmentation.max()
        boundary_mask = (1.-hmap) < self.boundary_threshold

        # Find new connected components:
        segmentation = vigra.analysis.labelVolume((segmentation * np.logical_not(boundary_mask)).astype('uint32'))

        # Intersect and relabel (to reduce the number of label_max):
        new_segmentation = np.where(boundary_mask, pixel_segm, segmentation)
        new_segmentation = vigra.analysis.relabelConsecutive(new_segmentation)[0]

        return new_segmentation
