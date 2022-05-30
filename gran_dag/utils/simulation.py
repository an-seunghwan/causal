#%%
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader

import numpy as np
import random
import igraph as ig
#%%
def sigmoid(z):
    return 1. / (1. + np.exp(-z))
#%%
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
#%%
def is_dag(W):
    """check DAGness
    Args:
        W (np.ndarray): weight matrix of graph

    Returns:
        Boolean: True if W is weight matrix of DAG
    """
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()
#%%
def simulate_dag(d, s0, graph_type):
    """Simulate random DAG with some expected number of edges
    Args:
        d (int): number of nodes
        s0 (int): expected number of edges
        graph_type (str): ER, SF, BP

    Returns:
        B (np.ndarray): d x d binary adjacency matrix of DAG
    """
    
    '''?????'''
    def _random_permutation(M):
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)
    
    def _graph_to_adjmat(G):
        # make igraph object to adjacency matrix 
        return np.array(G.get_adjacency().data)
    
    if graph_type == 'ER':
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
        B = _graph_to_adjmat(G)
    elif graph_type == 'BP':
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * d)
        G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    else:
        raise ValueError('unknown graph type')
    
    B_perm = _random_permutation(B)
    
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    
    return B_perm
#%%
def simulate_nonlinear_sem(B, n, sem_type, batch_size, noise_scale=None):
    """Simulate samples from nonlinear SEM.
    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        n (int): num of samples
        sem_type (str): mlp, mim, gp, gp-add
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones
        
    Returns:
        X (np.ndarray): [n, d] sample matrix
    """
    
    def _simulate_single_equation(x, scale):
        """
        input:
            x: [n, num of parents]
            scale: noise size
        output:
            x: [n]
        """
        
        z = np.random.normal(scale=scale, size=n)
        
        parent_size = x.shape[1]
        if parent_size == 0:
            return z
        
        weight_range = (0.5, 2.)
        low, high = weight_range
        
        if sem_type == "mlp":
            # sampling MLP weights
            hidden = 100
            W1 = np.random.uniform(low=low, high=high, size=[parent_size, hidden])
            W1[np.random.rand(W1.shape[0], W1.shape[1]) < 0.5] *= -1
            W2 = np.random.uniform(low=low, high=high, size=hidden)
            W2[np.random.rand(W2.shape[0]) < 0.5] *= -1
            x = sigmoid(x @ W1) @ W2 + z
        elif sem_type == "mim":
            W1 = np.random.uniform(low=low, high=high, size=parent_size)
            W1[np.random.rand(W1.shape[0]) < 0.5] *= -1
            W2 = np.random.uniform(low=low, high=high, size=parent_size)
            W2[np.random.rand(W2.shape[0]) < 0.5] *= -1
            W3 = np.random.uniform(low=low, high=high, size=parent_size)
            W3[np.random.rand(W3.shape[0]) < 0.5] *= -1
            x = np.tanh(x @ W1) + np.cos(x @ W2) + np.sin(x @ W3) + z
        elif sem_type == 'gp':
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor()
            x = gp.sample_y(x, random_state=None).flatten() + z
        elif sem_type == 'gp-add':
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor()
            x = sum([gp.sample_y(x[:, i, None], random_state=None).flatten()
                     for i in range(x.shape[1])]) + z
        else:
            raise ValueError('unknown sem type')
        return x
    
    d = B.shape[0]
    scale_vec = noise_scale if noise_scale else np.ones(d)
    X = np.zeros([n, d])
    G = ig.Graph.Adjacency(B.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], scale_vec[j]) 
    
    X = torch.FloatTensor(X)
    data = TensorDataset(X)
    data_loader = DataLoader(data, batch_size=batch_size)
    return data_loader
#%%
def count_accuracy(B_true, B_est):
    """Compute various accuracy metrics for B_est.
    reverse = predicted association exists in condition in opposite direction
    false positive = predicted association does not exist in condition
    
    Note that: Partial DAG is not considered!
    
    Args:
        B_true (np.ndarray): d x d ground truth binary adjacency graph, {0, 1}
        B_est (np.ndarray): d x d estimated binary adjacency graph, {0, 1}
        
    Returns:
        FDR: (reverse + false positive) / prediction 
        SHD: (undirected) extra + (undirected) missing + reverse
        nonzero: prediction positive
    """
    
    if not ((B_est == 0) | (B_est == 1)).all():
        raise ValueError('B_est should take value in {0,1}')
    if not is_dag(B_est):
        raise ValueError('B_est should be a DAG')

    d = B_true.shape[0]

    # linear index of nonzero weights
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])

    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    FDR = float(len(false_pos) + len(reverse)) / max(len(pred), 1)

    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    SHD = len(extra_lower) + len(missing_lower) + len(reverse)

    return {'FDR': FDR, 'SHD': SHD, 'nonzero': len(pred)}
#%%