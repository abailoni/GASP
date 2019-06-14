import sys
import os

HCI_HOME = "/home/abailoni_local/hci_home"
sys.path += [
    os.path.join(HCI_HOME, "python_libraries/nifty/python")]

import numpy as np

from GASP.segmentation import GaspFromAffinities, WatershedOnDistanceTransformFromAffinities

IMAGE_SHAPE = (10, 200, 200)
START_AGGLO_FROM_WSDT_SUPERPIXELS = True

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
random_affinities = np.random.uniform(size=(len(offsets),) + IMAGE_SHAPE).astype('float32')

# Run GASP:
if START_AGGLO_FROM_WSDT_SUPERPIXELS:
    # In this case the agglomeration is initialized with superpixels:
    # use additional option 'intersect_with_boundary_pixels' to break the SP along the boundaries
    # (see CREMI-experiments script for an example)
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

gasp_instance = GaspFromAffinities(offsets,
                                   superpixel_generator=superpixel_gen,
                                   run_GASP_kwargs=run_GASP_kwargs)
final_segmentation, runtime = gasp_instance(random_affinities)

print("Clustering took {} s".format(runtime))
