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
        graph_type (str): ER, SF
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
    return W.round(2)
#%%
#%%
def simulate_sem(W: np.ndarray, 
                n: int, 
                x_dim: int,
                nonlinear_type: str = 'nonlinear_1',
                sem_type: str = 'gauss', 
                noise_scale=None):
    """simulate samples from linear SEM with specified type of noise.
    Args:
        W (np.ndarray): d x d weighted adjacency matrix of DAG
        n (int): number of samples
        x_dim (int): dimension of each node variable
        sem_type (str): gauss, exp, gumbel, uniform, logistic, poisson
        nonlinear_type (str): nonlinear_1, nonlinear_2
        noise_scale (float): scale parameter of addictive noise, default one
    Returns:
        X (np.ndarray): n x d sample matrix, 
    """

    def _simulate_single_equation(x, w, scale):
        """
        x: [n, num of parents]
        w: [num of parents, ]
        h: n x 1
        """
        
        if nonlinear_type == 'nonlinear_1':
            h = np.cos(x + 1).dot(w)
        elif nonlinear_type == 'nonlinear_2':
            h = 2. * np.sin((x + 0.5).dot(w)) + (x + 0.5).dot(w)
        else:
            raise ValueError('unknown nonlinear type')
        
        if sem_type == 'gauss':
            z = scale * np.random.normal(size=n)
            h += z
        elif sem_type == 'exp':
            z = np.random.exponential(scale=scale, size=n)
            h += z
        elif sem_type == 'gumbel':
            z = np.random.gumbel(scale=scale, size=n)
            h += z
        elif sem_type == 'uniform':
            z = np.random.uniform(low=-scale, high=scale, size=n)
            h += z
        elif sem_type == 'logistic':
            h = np.random.binomial(1, sigmoid(h)) * 1.0
        elif sem_type == 'poisson':
            h = np.random.poisson(np.exp(h)) * 1.0
        else:
            raise ValueError('unknown sem type')
        return h
    
    d = W.shape[0]
    
    # all node (variable) and dimensions share the same noise scale
    if noise_scale is None:
        scale = 1.
    else:
        if not np.isscalar(noise_scale):
            raise ValueError('noise scale must be scalar!')
        scale = noise_scale
    
    if not is_dag(W):
        raise ValueError('W must be a DAG')
    
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    
    # first dimension for all node (variable)
    X = np.zeros((n, d, x_dim))
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j, 0] = _simulate_single_equation(X[:, parents, 0], W[parents, j], scale)
    
    # other dimenions
    if x_dim > 1:
        for i in range(1, x_dim):
            X[:, :, i] = np.random.normal(scale=scale, size=1) * X[:, :, 0]
            X[:, :, i] += np.random.normal(scale=scale, size=1)
            X[:, :, i] += np.random.normal(scale=scale, size=(n, d))
        X[:, :, 0] = np.random.normal(scale=scale, size=1) * X[:, :, 0]
        X[:, :, 0] += np.random.normal(scale=scale, size=1)
        X[:, :, 0] += np.random.normal(scale=scale, size=(n, d))
    return X
#%%
def load_data(config):
    W = simulate_dag(
        config["d"],
        config["degree"],
        config["graph_type"],
    )
    X = simulate_sem(
        W,
        config["n"],
        config["x_dim"],
        config["nonlinear_type"],
        config["sem_type"]
    )
    
    X = torch.FloatTensor(X)
    return X, W
    # data = TensorDataset(X)
    # data_loader = DataLoader(data, batch_size=config["batch_size"])
    # return data_loader, W
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
def main():
    n = 100
    d = 5
    degree = 4
    x_dim = 3
    # graph_type = "ER"
    # sem_type = "gauss"
    # nonlinear_type = "nonlinear_2"
    
    for graph_type in ['ER', 'SF']:
        W = simulate_dag(d, degree, graph_type)
        assert is_dag(W)
        assert W.shape == (d, d)
    print('graph_type passed!\n')
        
    for sem_type in ['gauss', 'exp', 'gumbel', 'uniform', 'logistic', 'poisson']:
        X = simulate_sem(W, n, x_dim, sem_type=sem_type)    
        assert X.shape == (n, d, x_dim)
    print('sem_type passed!\n')
    
    for nonlinear_type in ['nonlinear_1', 'nonlinear_2']:
        X = simulate_sem(W, n, x_dim, nonlinear_type=nonlinear_type)
        assert X.shape == (n, d, x_dim)
    print('nonlinear_type passed!\n')
        
    noise_scale = np.ones((d, x_dim - 1))
    try:
        X = simulate_sem(W, n, x_dim, noise_scale=noise_scale)
    except:
        print('noise_scale passed!\n')
#%%
if __name__ == '__main__':
    main()
#%%
"""FIXME"""
# def simulate_sem(W: np.ndarray, 
#                 n: int, 
#                 x_dim: int,
#                 nonlinear_type: str = 'nonlinear_1',
#                 sem_type: str = 'gauss', 
#                 noise_scale=None):
#     """simulate samples from linear SEM with specified type of noise.
#     Args:
#         W (np.ndarray): d x d weighted adjacency matrix of DAG
#         n (int): number of samples
#         x_dim (int): dimension of each node variable
#         sem_type (str): gauss, exp, gumbel, uniform, logistic, poisson
#         nonlinear_type (str): nonlinear_1, nonlinear_2
#         noise_scale (np.ndarray): scale parameter of addictive noise, default all ones
#     Returns:
#         X (np.ndarray): n x d sample matrix, 
#     """

#     def _simulate_single_equation(x, w, scale):
#         """
#         x: [n, num of parents, x_dim]
#         w: [num of parents, 1]
#         h: n x 1
#         """
#         if nonlinear_type == 'nonlinear_1':
#             h = w.T @ np.cos(x + 1)
#         elif nonlinear_type == 'nonlinear_2':
#             h = 2. * np.sin(w.T @ (x + 0.5)) + w.T @ (x + 0.5)
#         else:
#             raise ValueError('unknown nonlinear type')
        
#         if sem_type == 'gauss':
#             z = scale * np.random.normal(size=(n, 1, x_dim))
#             h += z
#         elif sem_type == 'exp':
#             z = np.concatenate([np.random.exponential(scale=s, size=(n, 1, 1)) for s in scale], axis=-1)
#             h += z
#         elif sem_type == 'gumbel':
#             z = np.concatenate([np.random.gumbel(scale=s, size=(n, 1, 1)) for s in scale], axis=-1)
#             h += z
#         elif sem_type == 'uniform':
#             z = np.concatenate([np.random.uniform(low=-s, high=s, size=(n, 1, 1)) for s in scale], axis=-1)
#             h += z
#         elif sem_type == 'logistic':
#             h = np.random.binomial(1, sigmoid(h)) * 1.0
#         elif sem_type == 'poisson':
#             h = np.random.poisson(np.exp(h)) * 1.0
#         else:
#             raise ValueError('unknown sem type')
#         return h
    
#     d = W.shape[0]
    
#     if noise_scale is None:
#         scale_vec = np.ones((d, x_dim))
#     elif np.isscalar(noise_scale):
#         scale_vec = noise_scale * np.ones((d, x_dim))
#     else:
#         if noise_scale.shape != (d, x_dim):
#             raise ValueError('noise scale shape must be (d, x_dim)')
#         scale_vec = noise_scale
    
#     if not is_dag(W):
#         raise ValueError('W must be a DAG')
    
#     G = ig.Graph.Weighted_Adjacency(W.tolist())
#     ordered_vertices = G.topological_sorting()
#     assert len(ordered_vertices) == d
    
#     X = np.zeros((n, d, x_dim))
#     for j in ordered_vertices:
#         parents = G.neighbors(j, mode=ig.IN)
#         X[:, [j], :] = _simulate_single_equation(X[:, parents], W[parents, j][..., None], scale_vec[j, :])
#     return X
#%%