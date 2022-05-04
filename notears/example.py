#%%
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import igraph as ig
import random
#%%
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

set_random_seed(0)
#%%
'''binary adj matrix of DAG'''
d, s0, graph_type = 5, 5, 'ER'

# Erdos-Renyi
G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)

def _graph_to_adjmat(G):
    return np.array(G.get_adjacency().data)
B_und = _graph_to_adjmat(G_und)

def _random_permutation(M):
    # np.random.permutation permutes first axis only
    P = np.random.permutation(np.eye(M.shape[0]))
    return P.T @ M @ P
def _random_acyclic_orientation(B_und):
    return np.tril(_random_permutation(B_und), k=-1)
B = _random_acyclic_orientation(B_und)

B = _random_permutation(B)
ig.Graph.Adjacency(B.tolist()).is_dag() # check DAGness
#%%
'''weighted adj matrix of DAG'''
w_ranges=((-2.0, -0.5), (0.5, 2.0))

W = np.zeros(B.shape)
S = np.random.randint(len(w_ranges), size=B.shape)  # which range
for i, (low, high) in enumerate(w_ranges):
    U = np.random.uniform(low=low, high=high, size=B.shape)
    W += B * (S == i) * U
W = np.round(W, 2)
#%%
'''visualize weighted adj matrix of DAG'''
plt.figure(figsize=(6, 6))
G = nx.from_numpy_matrix(W, create_using=nx.DiGraph)
layout = nx.circular_layout(G)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw(G, layout, 
        with_labels=True, 
        font_size=20,
        font_weight='bold',
        arrowsize=30,
        node_size=1000)
nx.draw_networkx_edge_labels(G, 
                             pos=layout, 
                             edge_labels=labels, 
                             font_weight='bold',
                             font_size=15)
plt.savefig(
    "./assets/DAG_example.png",
    dpi=100,
    bbox_inches="tight",
    pad_inches=0.1,
)
plt.show()
plt.close()
#%%