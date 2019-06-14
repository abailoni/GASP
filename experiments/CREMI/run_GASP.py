import sys
import os

# TODO: remove
HCI_HOME = "/home/abailoni_local/hci_home"
sys.path += [
    os.path.join(HCI_HOME, "python_libraries/nifty/python")]

import argparse
import os
import h5py

from GASP.segmentation import GaspFromAffinities, WatershedOnDistanceTransformFromAffinities


def run_GASP_on_affinities(affinities,
                           offsets,
                           linkage_criteria="average",
                           add_cannot_link_constraints=False):
    # Prepare graph pre-processor or superpixel generator:
    # a WSDT segmentation is intersected with connected components
    boundary_kwargs = {
        'boundary_threshold': 0.5,
        'used_offsets': [0, 1, 2, 4, 5, 7, 8],
        'offset_weights': [1., 1., 1., 1., 1., 0.9, 0.9]
    }
    superpixel_gen = WatershedOnDistanceTransformFromAffinities(offsets,
                                                                threshold=0.4,
                                                                min_segment_size=20,
                                                                preserve_membrane=True,
                                                                sigma_seeds=0.1,
                                                                stacked_2d=True,
                                                                intersect_with_boundary_pixels=True,
                                                                boundary_pixels_kwargs=boundary_kwargs,
                                                                )
    run_GASP_kwargs = {'linkage_criteria': linkage_criteria,
                       'add_cannot_link_constraints': add_cannot_link_constraints}

    gasp_instance = GaspFromAffinities(offsets,
                                       superpixel_generator=superpixel_gen,
                                       run_GASP_kwargs=run_GASP_kwargs)
    final_segmentation, runtime = gasp_instance(affinities)
    print("Clustering took {} s".format(runtime))

    return final_segmentation


def load_cremi_dataset(cremi_folder_path, sample):
    # TODO: define train-crops, move this folder to somewhere else...
    allowed_samples = ["A", "B", "C", "A+", "B+", "C+"]
    assert sample in allowed_samples, "The accepted cremi samples should be chose among {}".format(allowed_samples)

    is_training_sample = "+" not in sample
    if is_training_sample:
        cremi_folder_path = os.path.join(cremi_folder_path, "train")
    else:
        cremi_folder_path = os.path.join(cremi_folder_path, "test")
    assert os.path.exists(cremi_dataset_folder), cremi_dataset_folder

    with h5py.File(os.path.join(cremi_folder_path, "affinities", "sample_{}.h5")) as f:
        affinities = f['data'][:]
    with h5py.File(os.path.join(cremi_folder_path, "data", "sample_{}.h5")) as f:
        raw = f['raw_data'][:]
        if is_training_sample:
            gt = f['gt'][:]
    return raw, affinities, gt


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cremi_dataset_folder', type=str,
                        help='path to directory downloaded from ....')  # TODO: path!
    parser.add_argument('samples', nargs='+', type=str, default=["A", "B", "C"],
                        help='Which CREMI samples should be processed (A, B or C for training, A+, B+ and C+ for test)')
    parser.add_argument('--linkage_criteria', type=str, default='average',
                        help='Update rule used by GASP')
    parser.add_argument('--add_cannot_link_constraints', type=str2bool, default='false',
                        help='Add cannot-link constraints during GASP agglomeration')

    args = parser.parse_args()

    offsets = [
        # Direct 3D neighborhood:
        [-1, 0, 0], [0, -1, 0], [0, 0, -1],
        # Additional long-range edges:
        [-2, 0, 0],
        [0, -3, 0],
        [0, 0, -3],
        [-3, 0, 0],
        [0, -9, 0],
        [0, 0, -9],
        [-4, 0, 0],
        [0, -27, 0],
        [0, 0, -27]
    ]

    cremi_dataset_folder = args.cremi_dataset_folder
    assert os.path.exists(cremi_dataset_folder), cremi_dataset_folder

    for sample in args.samples:
        _, affinities, _ = load_cremi_dataset(cremi_dataset_folder, sample)
        final_segm = run_GASP_on_affinities(affinities,
                                            offsets,
                                            linkage_criteria=args.linkage_criteria,
                                            add_cannot_link_constraints=args.add_cannot_link_constraints)
        # TODO: write final_segm to file
