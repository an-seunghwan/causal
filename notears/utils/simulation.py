#%%
import numpy as np
#%%
def _graph_to_adjmat(G):
    return np.array(G.get_adjacency().data)

def _random_permutation(M):
    P = np.random.permutation(np.eye(M.shape[0]))
    return P.T @ M @ P

def _random_acyclic_orientation(B_und):
    return np.tril(_random_permutation(B_und), k=-1)
#%%