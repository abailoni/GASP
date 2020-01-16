import numpy as np
import sys
sys.path += ["/home/abailoni_local/hci_home/pyCharm_projects/longRangeAgglo/"]
sys.path += ["/home/abailoni_local/hci_home/pyCharm_projects/GASP/"]
import long_range_compare

from GASP.segmentation import GaspFromAffinities, WatershedOnDistanceTransformFromAffinities

IMAGE_SHAPE = (10, 50, 50)
START_AGGLO_FROM_WSDT_SUPERPIXELS = False

offsets = [
    # Direct 3D neighborhood:
    [-1, 0, 0], [0, -1, 0], [0, 0, -1],
    # Long-range connections:
    [-1, -1, -1],
    [-1, 1, 1],
    [-1, -1, 1],
    [-1, 1, -1],
    [0, -9, 0],
    [0, 0, -9],
    [0, -9, -9],
    [0, 9, -9],
    [0, -9, -4],
    [0, -4, -9],
    [0, 4, -9],
    [0, 9, -4],
    [0, -27, 0],
    [0, 0, -27]]

# Generate some random affinities:
random_affinities = np.zeros((len(offsets),) + IMAGE_SHAPE).astype('float32')

# Run GASP:
if START_AGGLO_FROM_WSDT_SUPERPIXELS:
    # In this case the agglomeration is initialized with superpixels:
    # use additional option 'intersect_with_boundary_pixels' to break the SP along the boundaries
    # (see CREMI-experiments script for an example)
    # FIXME: if superpixels are used, the stacked dimension should be the first one
    superpixel_gen = WatershedOnDistanceTransformFromAffinities(offsets,
                                                                threshold=0.4,
                                                                min_segment_size=20,
                                                                preserve_membrane=True,
                                                                sigma_seeds=0.1,
                                                                stacked_2d=True,
                                                                )
else:
    superpixel_gen = None

run_GASP_kwargs = {'linkage_criteria': 'average',
                   'add_cannot_link_constraints': False}

class BackgroundLabelSuperpixelGenerator(object):
    def __call__(self, affinities, foreground_mask):
        pixel_segm = np.arange(np.prod(foreground_mask.shape), dtype='uint64').reshape(foreground_mask.shape) + 1
        return (pixel_segm * foreground_mask).astype('int64') - 1


superpixel_gen = BackgroundLabelSuperpixelGenerator()

gasp_instance = GaspFromAffinities(offsets,
                                   superpixel_generator=superpixel_gen,
                                   run_GASP_kwargs=run_GASP_kwargs)
mask = np.ones_like(random_affinities[0])
mask[:5] = 0
final_segmentation, runtime = gasp_instance(random_affinities, mask)

print("Clustering took {} s".format(runtime))


