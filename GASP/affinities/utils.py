import numpy as np
import vigra
from concurrent import futures

from nifty.graph import rag as nrag

def from_affinities_to_hmap(affinities, offsets, used_offsets=None, offset_weights=None):
    """
    :param affinities: Merge probabilities (channels,z,x,y) or (channels,x,y)
    :param offsets: np.array
    :param used_offsets: list of offset indices
    :param offset_weights: list of weights
    :return:
    """
    if isinstance(offsets, list):
        offsets = np.array(offsets)

    inverted_affs = 1. - affinities
    if used_offsets is None:
        used_offsets = range(offsets.shape[0])
    if offset_weights is None:
        offset_weights = [1.0 for _ in range(len(used_offsets))]
    assert len(used_offsets)==len(offset_weights)
    rolled_affs = []
    for i, offs_idx in enumerate(used_offsets):
        offset = offsets[offs_idx]
        shifts = tuple([int(off/2) for off in offset])

        padding = [[0, 0] for _ in range(len(shifts))]
        for ax, shf in enumerate(shifts):
            if shf < 0:
                padding[ax][1] = -shf
            elif shf > 0:
                padding[ax][0] = shf
        padded_inverted_affs = np.pad(inverted_affs, pad_width=((0, 0),) + tuple(padding), mode='constant')
        crop_slices = tuple(slice(padding[ax][0], padded_inverted_affs.shape[ax+1] - padding[ax][1]) for ax in range(3))
        rolled_affs.append(np.roll(padded_inverted_affs[offs_idx], shifts, axis=(0,1,2))[crop_slices] * offset_weights[i])
    prob_map = np.stack(rolled_affs).max(axis=0)

    return prob_map


def probs_to_costs(probs,
                   beta=.5,
                   weighting_scheme=None,
                   rag=None,
                   segmentation=None,
                   weight=16.):
    """
    :param probs: expected a probability map (0.0 merge or 1.0 split)
    :param beta: bias factor (with 1.0 everything is repulsive, with 0. everything is attractive)
    """
    p_min = 0.001
    p_max = 1. - p_min
    # Costs: positive (merge), negative (split)
    costs = (p_max - p_min) * probs + p_min

    # probabilities to energies, second term is boundary bias
    costs = np.log((1. - costs) / costs) + np.log((1. - beta) / beta)

    if weighting_scheme is not None:
        assert rag is not None
        assert weighting_scheme in ('xyz', 'z', 'all')
        assert segmentation is not None
        shape = segmentation.shape
        fake_data = np.zeros(shape, dtype='float32')
        edge_sizes = nrag.accumulateEdgeMeanAndLength(rag, fake_data)[:, 1]

        if weighting_scheme == 'all':
            w = weight * edge_sizes / edge_sizes.max()
        else:
            raise NotImplementedError("Weighting scheme not implemented")
        costs *= w

    return costs



def size_filter(hmap, seg, threshold):
    segments, counts = np.unique(seg, return_counts=True)
    mask = np.ma.masked_array(seg, np.in1d(seg, segments[counts < threshold])).mask
    filtered = seg.copy()
    filtered[mask] = 0
    filtered, _ = vigra.analysis.watershedsNew(hmap, seeds=filtered.astype("uint32"))
    filtered, max_label, _ = vigra.analysis.relabelConsecutive(filtered, start_label=1)
    return filtered, max_label


def superpixel_stacked(hmap, sp2d_fu, n_threads):
    segmentation = np.zeros(hmap.shape, dtype='uint32')

    def run_sp_2d(z):
        seg, off = sp2d_fu(hmap[z])
        segmentation[z] = seg
        return off + 1

    with futures.ThreadPoolExecutor(max_workers=n_threads) as tp:
        tasks = [tp.submit(run_sp_2d, z) for z in range(len(segmentation))]
        offsets = [t.result() for t in tasks]

    offsets = np.roll(offsets, 1)
    offsets[0] = 0
    offsets = np.cumsum(offsets).astype('uint32')
    segmentation += offsets[:, None, None]
    return segmentation, segmentation.max()
