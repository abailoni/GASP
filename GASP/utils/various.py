import numpy as np


def check_offsets(offsets):
    if isinstance(offsets, list):
        offsets = np.array(offsets)
    else:
        assert isinstance(offsets, np.ndarray)
    assert offsets.ndim == 2
    return offsets
