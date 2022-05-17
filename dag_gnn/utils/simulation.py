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
    return W
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