import sys
import os

HCI_HOME = "/home/abailoni_local/hci_home"
sys.path += [
    os.path.join(HCI_HOME, "python_libraries/nifty/python")]

import numpy as np

from nifty.graph import UndirectedGraph
from GASP.segmentation import run_GASP

# Generate a random graph:
nb_nodes, nb_edges = 1000, 3000
graph = UndirectedGraph(nb_nodes)
random_edges = np.random.randint(0, nb_nodes - 1, size=(nb_edges, 2))
# Delete self-loop edges:
# FIXME: something is going wrong, double edges?
check = random_edges[:,0] == random_edges[:,1]
print(check.sum(), random_edges.shape)
random_edges = np.delete(random_edges, np.argwhere(check)[0], axis=0)
graph.insertEdges(random_edges)
print(random_edges.shape)

# Run GASP:
random_signed_weights = np.random.uniform(-1., 1., size=graph.numberOfEdges)
final_node_labels, runtime = run_GASP(graph,
                                      random_signed_weights,
                                      linkage_criteria='average',
                                      add_cannot_link_constraints=False,
                                      verbose=True,
                                      print_every=100)

print("Clustering took {} s".format(runtime))
