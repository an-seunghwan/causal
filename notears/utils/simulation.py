#%%
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
def simulate_parameter(B, w_ranges=((-2.0, -0.5), (0.5, 2.0))):
    """simulate SEM parameters for a DAG
    Args:
        B (np.ndarray): d x d binary adjacency matrix of DAG
        w_ranges (tuple): disjoint weight ranges

    Returns:
        W (np.ndarray): d x d weighted adjacency matrix of DAG
    """
    W = np.zeros(B.shape)
    S = np.random.randint(len(w_ranges), size=B.shape)  # choice of range
    for i, (low, high) in enumerate(w_ranges):
        U = np.random.uniform(low=low, high=high, size=B.shape)
        W += B * (S == i) * U
    return W.round(2)
#%%
def simulate_linear_sem(W, n, sem_type, noise_scale=None, normalize=True):
    """simulate samples from linear SEM with specified type of noise.
    Args:
        W (np.ndarray): d x d weighted adjacency matrix of DAG
        n (int): number of samples, n = inf mimics population risk
        sem_type (str): gauss, exp, gumbel, uniform, logistic, poisson
        noise_scale (np.ndarray): scale parameter of addictive noise, default all ones
        normalize (bool): If True, normalize simulated dataset

    Returns:
        X (np.ndarray): n x d sample matrix, 
            if n == inf: d x d
    """

    def _simulate_single_equation(X, w, scale):
        """
        X: n x num of parents
        w: num of parents x 1
        x: n x 1
        """
        if sem_type == 'gauss':
            z = np.random.normal(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'exp':
            z = np.random.exponential(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'gumbel':
            z = np.random.gumbel(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'uniform':
            z = np.random.uniform(low=-scale, high=scale, size=n)
            x = X @ w + z
        elif sem_type == 'logistic':
            x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
        elif sem_type == 'poisson':
            x = np.random.poisson(np.exp(X @ w)) * 1.0
        else:
            raise ValueError('unknown sem type')
        return x
    
    d = W.shape[0]
    
    if noise_scale is None:
        scale_vec = np.ones(d)
    elif np.isscalar(noise_scale):
        scale_vec = noise_scale * np.ones(d)
    else:
        if len(noise_scale) != d:
            raise ValueError('noise scale must be a scalar or has length d')
        scale_vec = noise_scale
    
    if not is_dag(W):
        raise ValueError('W must be a DAG')
    
    '''?????'''
    if np.isinf(n): # population risk for linear gauss SEM
        if sem_type == 'gauss':
            # make 1/d X @ X.T = true covariance matrix
            X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
            return X
        else:
            raise ValueError('population risk not available')
    
    # empirical risk
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    
    X = np.zeros((n, d))
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j])
    
    if normalize:
        X = X - np.mean(X, axis=0, keepdims=True) # normalize
    return X
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