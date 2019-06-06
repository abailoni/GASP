class GaspFromAffinities(object):
    def __init__(self,
                 offsets,
                 superpixel_generator=None,
                 offsets_probabilities=None,
                 used_offsets=None,
                 offsets_weights=None,
                 n_threads=1,
                 invert_affinities=False,
                 GASP_kwargs=None,
                 mask_used_edges=None):
        self.offsets = offsets
        self.superpixel_generator = superpixel_generator
        self.offsets_probabilities = offsets_probabilities
        self.used_offsets = used_offsets
        self.offsets_weights = offsets_weights
        self.n_threads = n_threads
        self.invert_affinities = invert_affinities
        self.GASP_kwargs = GASP_kwargs
        self.mask_used_edges = mask_used_edges



