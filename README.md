# GASP
Generalized Algorithm for Signed graph Partitioning


## Installation
#### On Linux
- Create conda environment and install the `nifty` and `vigra` packages with: `conda create -n GASP -c abailoni -c conda-forge nifty vigra`. The source code for the `nifty` package can also be found here: https://github.com/abailoni/nifty
- Activate the environment: `source activate GASP`
- Clone this repository: `git clone https://github.com/abailoni/GASP.git`
- `cd ./GASP`
- Install the package: `python setup.py install`

<!--
    - Dependencies to check: h5py
    - Add arXiv link
    -
-->

## CREMI experiments
You will be soon able to download the CREMI data and the affinities predicted by our model.

To run GASP with our predictions, go to the `experiments/CREMI` folder and run the `run_GASP` script:

    python run_GASP.py /path/to/cremi/dataset/folder --samples A B C --linkage_criteria average --add_cannot_link_constraints false

Samples `A`, `B` and `C` correspond to training data. Samples `A+`, `B+` and `C+` correspond to test data. See more details about `linkage_criteria` below.
## Description of the package and how to use it
#### Examples
In the folder `examples` there are some scripts to run the GASP algorithm directly on a graph or on affinities generated from an image.

### Running GASP on a graph
The main function to run GASP on a graph (that can be built using the `nifty` package) is given by `from GASP.segmentation import run_GASP`:

```python
run_GASP(
        graph,
        signed_edge_weights,
        linkage_criteria='mean',
        add_cannot_link_constraints=False,
        edge_sizes=None,
        is_mergeable_edge=None,
        use_efficient_implementations=True,
        verbose=False,
        linkage_criteria_kwargs=None,
        print_every=100000)
```

with the following parameters:

- `graph` :

  Instance of a `nifty.graph`, e.g. `nifty.graph.UndirectedGraph`, `nifty.graph.undirectedLongRangeGridGraph` or `nifty.graph.rag.gridRag`

- `signed_edge_weights` : numpy.array(float)

  Array with shape (nb_graph_edges, ). Attractive weights are positive; repulsive weights are negative.

- `linkage_criteria` : str (default `mean`)

  Specifies the linkage criteria / update rule used during agglomeration

    - `mean`, `average`, `avg`
    - `max`, `single_linkage`
    - `min`, `complete_linkage`
    - `mutex_watershed`, `abs_max`
    - `sum`
    - `quantile`, `rank` keeps statistics in a histogram, with parameters:
        - `q` : float (default 0.5 equivalent to the median)
        - `numberOfBins`: int (default: 40)
    - `generalized_mean`, `gmean` (https://en.wikipedia.org/wiki/Generalized_mean) with parameters:
        - `p` : float (default: 1.0)
    - `smooth_max`, `smax` (https://en.wikipedia.org/wiki/Smooth_maximum) with parameters:
        - `p` : float (default: 0.0)

- `add_cannot_link_constraints` : bool

- `edge_sizes` : numpy.array(float) with shape (nb_graph_edges, )

  Depending on the linkage criteria, they can be used during the agglomeration to weight differently
        the edges  (e.g. with sum or avg linkage criteria). Commonly used with regionAdjGraphs when edges
        represent boundaries of different length between segments / super-pixels. By default, all edges have
        the same weighting.

- `is_mergeable_edge` : numpy.array(bool) with shape (nb_graph_edges, )

    Specifies if an edge can be merged or not. Sometimes some edges represent direct-neighbor relations
        and others describe long-range connections. If a long-range connection /edge is assigned to
        `is_mergeable_edge = False`, then the two associated nodes are not merged until they become
        direct neighbors and they get connected in the image-plane.
        By default all edges are mergeable.

- `use_efficient_implementations` : bool (default: True)

   In the following special cases, alternative efficient implementations are used:

    - `abs_max` criteria: Mutex Watershed (https://github.com/hci-unihd/mutex-watershed.git)
    - `max` criteria without cannot-link constraints: maximum spanning tree

- `verbose` : bool (default: False)

- `linkage_criteria_kwargs` : dict

    Additional optional parameters passed to the chosen linkage criteria (see previous list)

- `print_every` : int (default: 100000)

     After how many agglomeration iteration to print in verbose mode


### Image segmentation with GASP
For more details about it, see example `examples/run_GASP_from_affinities.py` and the docstrings of the class `GASP.segmentation.GaspFromAffinities`:

```python
class GaspFromAffinities(object):
    def __init__(self,
                 offsets,
                 beta_bias=0.5,
                 superpixel_generator=None,
                 run_GASP_kwargs=None,
                 n_threads=1,
                 verbose=False,
                 invert_affinities=False,
                 offsets_probabilities=None,
                 use_logarithmic_weights=False,
                 used_offsets=None,
                 offsets_weights=None):
        ...

     def __call__(self, affinities, *args_superpixel_gen):
        ...
```
