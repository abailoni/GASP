import sys
import os

HCI_HOME = "/home/abailoni_local/hci_home"
sys.path += [
    os.path.join(HCI_HOME, "python_libraries/nifty/python"),
    os.path.join(HCI_HOME, "pyCharm_projects/segmfriends"),
]

import numpy as np

from nifty.graph import UndirectedGraph
from GASP.segmentation import run_GASP

# --------------------------
# Generate a random graph:
# --------------------------
nb_nodes, nb_edges = 7000, 10000
graph = UndirectedGraph(nb_nodes)
# Generate some random edges (but avoid self-loop):
random_edges = np.random.randint(0, nb_nodes - 1, size=(nb_edges, 2))
check = random_edges[:,0] == random_edges[:,1]
if any(check):
    random_edges = np.delete(random_edges, np.argwhere(check), axis=0)
# Add them to the graph:
graph.insertEdges(random_edges)
# Generate some random (signed) weights:
random_signed_weights = np.random.uniform(-1., 1., size=graph.numberOfEdges)

# --------------------------
# Run GASP:
# --------------------------
final_node_labels, runtime = run_GASP(graph,
                                      random_signed_weights,
                                      linkage_criteria='average',
                                      add_cannot_link_constraints=False,
                                      verbose=False,
                                      print_every=100)

print("Clustering took {} s".format(runtime))
