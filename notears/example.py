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

set_random_seed(5)
#%%
'''binary adj matrix of DAG'''
n, d, s0, graph_type = 100, 5, 5, 'ER'

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
# w_ranges=((-2.0, -0.5), (0.5, 2.0))
w_ranges=(-1, 1)

W = np.zeros(B.shape)
S = np.random.randint(len(w_ranges), size=B.shape)  # which range
# for i, (low, high) in enumerate(w_ranges):
for i, val in enumerate(w_ranges):
    # U = np.random.uniform(low=low, high=high, size=B.shape)
    U = np.ones(B.shape) * val
    W += B * (S == i) * U
# W = np.round(W, 2)
W = W.astype(int)
np.savetxt('./assets/W_example.csv', W, delimiter=',')
#%%
'''visualize weighted adj matrix of DAG'''
plt.figure(figsize=(6, 6))
G = nx.from_numpy_matrix(W, create_using=nx.DiGraph)
layout = nx.planar_layout(G)
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
'''heatmap'''
fig = plt.figure(figsize=(5, 4))
plt.pcolor(W, cmap='coolwarm', edgecolor='black')
plt.gca().invert_yaxis()
plt.colorbar()
plt.savefig(
    "./assets/DAG_example_heatmap.png",
    dpi=100,
    bbox_inches="tight",
    pad_inches=0.1,
)
plt.show()
plt.close()
#%%
'''simulate dataset'''
d = W.shape[0]
scale_vec = np.ones(d) # noise scale (standard deviation)

G = ig.Graph.Weighted_Adjacency(W.tolist())
ordered_vertices = G.topological_sorting()

X = np.zeros([n, d])
for j in ordered_vertices:
    parents = G.neighbors(j, mode=ig.IN)
    z = np.random.normal(scale=scale_vec[j], size=n)
    x = X[:, parents] @ W[parents, j] + z
    X[:, j] = x
np.savetxt('./assets/X_example.csv', X, delimiter=',')
#%%