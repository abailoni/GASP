import numpy as np


def check_offsets(offsets):
    if isinstance(offsets, (list, tuple)):
        offsets = np.array(offsets)
    else:
        assert isinstance(offsets, np.ndarray)
    assert offsets.ndim == 2
    return offsets


def find_indices_direct_neighbors_in_offsets(offsets):
    offsets = check_offsets(offsets)
    indices_dir_neighbor = []
    is_dir_neighbor = np.empty(offsets.shape[0], dtype='bool')
    for i, off in enumerate(offsets):
        is_dir_neighbor[i] = (np.abs(off).sum() == 1)
        if np.abs(off).sum() == 1:
            indices_dir_neighbor.append(i)
    return is_dir_neighbor, indices_dir_neighbor


def parse_data_slice(data_slice):
    """Parse a dataslice as a list of slice objects."""
    if data_slice is None:
        return data_slice
    elif isinstance(data_slice, (list, tuple)) and \
            all([isinstance(_slice, slice) for _slice in data_slice]):
        return list(data_slice)
    else:
        assert isinstance(data_slice, str)
    # Get rid of whitespace
    data_slice = data_slice.replace(' ', '')
    # Split by commas
    dim_slices = data_slice.split(',')
    # Build slice objects
    slices = []
    for dim_slice in dim_slices:
        indices = dim_slice.split(':')
        if len(indices) == 2:
            start, stop, step = indices[0], indices[1], None
        elif len(indices) == 3:
            start, stop, step = indices
        else:
            raise RuntimeError
        # Convert to ints
        start = int(start) if start != '' else None
        stop = int(stop) if stop != '' else None
        step = int(step) if step is not None and step != '' else None
        # Build slices
        slices.append(slice(start, stop, step))
    # Done.
    return tuple(slices)
