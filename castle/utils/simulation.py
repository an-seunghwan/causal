#%%
# """
# code from:
# https://github.com/vanderschaarlab/mlforhealthlabpub/blob/main/alg/castle/utils.py
# """
#%%
import torch
import pandas as pd
import numpy as np
import random
import igraph as ig
import networkx as nx
#%%
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
#%%
def sigmoid(z):
    return 1. / (1. + np.exp(-z))
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
                 graph_type: str):
    """Simulate random DAG with some expected number of edges
    Args:
        d (int): number of nodes
        degree (float): expected node degree (= in + out)
        graph_type (str): ER, SF
    Returns:
        B_perm (np.ndarray): d x d adjacency matrix of DAG
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
    
    return B_perm
#%%
def simulate_nonlinear_sem(B, n, sem_type, noise_scale=None):
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
    return X, G, ordered_vertices
#%%
# def random_dag(nodes, edges):
#     """Generate a random Directed Acyclic Graph (DAG) with a given number of nodes and edges."""
#     G = nx.DiGraph()
#     for i in range(nodes):
#         G.add_node(i)
#     while edges > 0:
#         a = random.randint(0, nodes-1)
#         b = a
#         while b == a:
#             b = random.randint(0,nodes-1)
#         G.add_edge(a,b)
#         if nx.is_directed_acyclic_graph(G):
#             edges -= 1
#         else:
#             # we closed a loop!
#             G.remove_edge(a,b)
#     return G
# #%%
# def swap_cols(df, a, b):
#     df = df.rename(columns = {a : 'temp'})
#     df = df.rename(columns = {b : a})
#     return df.rename(columns = {'temp' : b})
# def swap_nodes(G, a, b):
#     newG = nx.relabel_nodes(G, {a : 'temp'})
#     newG = nx.relabel_nodes(newG, {b : a})
#     return nx.relabel_nodes(newG, {'temp' : b})
# #%%
# # This function generates data according to a DAG provided in list_vertex and list_edges with mean and variance as input
# # It will apply a perturbation at each node provided in perturb.
# def gen_data_nonlinear(G, mean = 0, var = 1, SIZE = 10000, perturb = [], sigmoid = True):
#     list_edges = G.edges()
#     list_vertex = G.nodes()

#     order = []
#     for ts in nx.algorithms.dag.topological_sort(G):
#         order.append(ts)

#     g = []
#     for v in list_vertex:
#         if v in perturb:
#             g.append(np.random.normal(mean,var,SIZE))
#             print("perturbing ", v, "with mean var = ", mean, var)
#         else:
#             g.append(np.random.normal(0, 1, SIZE))

#     for o in order:
#         for edge in list_edges:
#             if o == edge[1]: # if there is an edge into this node
#                 if sigmoid:
#                     g[edge[1]] += 1 / (1 + np.exp(-g[edge[0]]))
#                 else:   
#                     g[edge[1]] += np.square(g[edge[0]])
#     g = np.swapaxes(g,0,1)
#     return pd.DataFrame(g, columns = list(map(str, list_vertex)))
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
        # raise ValueError('B_est should be a DAG')
        pred = np.flatnonzero(B_est == 1)
        return {'nonzero': len(pred)}

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
    # n = 1000
    # nodes = 10
    # edges = 4
    
    # G = random_dag(nodes, edges)
    # X = gen_data_nonlinear(G, SIZE=n, sigmoid=True)
    # assert X.shape == (n, nodes)
    # print('data generation passed!\n')
    
    n = 100
    d = 5
    s0 = 4
    # graph_type = "ER"
    # sem_type = "gauss"
    # nonlinear_type = "nonlinear_2"
    
    for graph_type in ['ER', 'SF', 'BP']:
        W = simulate_dag(d, s0, graph_type)
        assert is_dag(W)
        assert W.shape == (d, d)
    print('graph_type passed!\n')
        
    for sem_type in ['mlp', 'mim', 'gp', 'gp-add']:
        X, G, ordered_vertices = simulate_nonlinear_sem(W, n, sem_type=sem_type)    
        assert X.shape == (n, d)
    print('sem_type passed!\n')
#%%
if __name__ == '__main__':
    main()
#%%