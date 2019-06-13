import sys
import os

HCI_HOME = "/home/abailoni_local/hci_home"
sys.path += [
    os.path.join(HCI_HOME, "python_libraries/nifty/python")]

import numpy as np

from GASP.segmentation import GaspFromAffinities

offsets = [[-1, 0, 0],
           [0, -1, 0],
           [0, 0, -1],
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
IMAGE_SHAPE = (10, 200, 200)
random_affinities = np.random.uniform(size=(len(offsets),) + IMAGE_SHAPE)


# Run GASP:
run_GASP_kwargs = {'linkage_criteria': 'average',
                   'add_cannot_link_constraints': False}
gasp_instance = GaspFromAffinities(offsets,
                                   run_GASP_kwargs=run_GASP_kwargs)
final_segmentation, runtime = gasp_instance(random_affinities)

print("Clustering took {} s".format(runtime))
