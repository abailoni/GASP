import unittest

import numpy as np

from GASP.segmentation import GaspFromAffinities, WatershedOnDistanceTransformFromAffinities

class run_GASP_from_pixel_affinities(unittest.TestCase):
    def test_GASP_from_random_affinities_from_superpixels(self):
        IMAGE_SHAPE = (10, 40, 40)
        START_AGGLO_FROM_WSDT_SUPERPIXELS = True

        offsets = [
            # Direct 3D neighborhood:
            [-1, 0, 0], [0, -1, 0], [0, 0, -1],
            # Long-range connections:
            [-1, -1, -1]]

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

        run_GASP_kwargs = {'linkage_criteria': 'mutex_watershed',
                           'add_cannot_link_constraints': False}

        gasp_instance = GaspFromAffinities(offsets,
                                           superpixel_generator=superpixel_gen,
                                           run_GASP_kwargs=run_GASP_kwargs)
        final_segmentation, runtime = gasp_instance(random_affinities)

        # TODO: actually assert something!

    def test_GASP_from_random_affinities_from_pixels(self):
        IMAGE_SHAPE = (10, 40, 40)
        START_AGGLO_FROM_WSDT_SUPERPIXELS = False

        offsets = [
            # Direct 3D neighborhood:
            [-1, 0, 0], [0, -1, 0], [0, 0, -1],
            # Long-range connections:
            [-1, -1, -1]]

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

        # TODO: actually assert something!


if __name__ == '__main__':
    unittest.main()
