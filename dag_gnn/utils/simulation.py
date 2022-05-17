#%%
import numpy as np
import random
import igraph as ig
#%%
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
#%%
def is_dag(W: np.ndarray):
    """check DAGness
    Args:
        W: weight matrix of graph
    Returns:
        Boolean: True if W is weight matrix of DAG
    """
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()
#%%
def simulate_dag(d: int, 
                 degree: float, 
                 graph_type: str,
                 w_ranges: tuple = ((-2.0, -0.5), (0.5, 2.0))):
    """Simulate random DAG with some expected number of edges
    Args:
        d (int): number of nodes
        degree (float): expected node degree (= in + out)
        graph_type (str): ER, SF, BP
        w_ranges (tuple): disjoint weight ranges
    Returns:
        W (np.ndarray): d x d weighted adjacency matrix of DAG
    """
    
    def _random_permutation(M):
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P
    
    if graph_type == 'ER': # Erdos-Renyi
        prob = float(degree) / (d - 1)
        # select nodes which are goint to be connected (with probability prob)
        B = np.tril((np.random.rand(d, d) < prob).astype(float), k=-1) # lower triangular
    elif graph_type == 'SF': # Scale-free, Barabasi-Albert
        m = int(round(degree / 2))
        B = np.zeros([d, d])
        bag = [0]
        for i in range(1, d):
            dest = np.random.choice(bag, size=m)
            for j in dest:
                B[i, j] = 1
            bag.append(i)
            bag.extend(dest)
    else:
        raise ValueError('unknown graph type')
    
    W = np.zeros(B.shape)
    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    
    S = np.random.randint(len(w_ranges), size=B.shape)  # choice of range
    for i, (low, high) in enumerate(w_ranges):
        U = np.random.uniform(low=low, high=high, size=B.shape)
        W += B_perm * (S == i) * U
    return W
#%%