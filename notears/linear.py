#%%
import os
from pydoc import visiblename 
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

import networkx as nx
import igraph as ig

from utils.simulation import (
    set_random_seed,
    simulate_dag,
    simulate_parameter,
    simulate_linear_sem,
)

from utils.viz import (
    viz_graph
)
#%%
params = {
    # "neptune": True, # True if you use neptune.ai
    
    "seed": 10,
    "n": 200,
    "d": 5,
    "s0": 5,
    "graph_type": 'ER',
    "sem_type": 'gauss',
    
    "rho": 1., # initial value
    "alpha": 0., # initial value
    "h": np.inf, # initial value
    
    "loss_type": 'l2',
    "max_iter": 100, 
    "h_tol": 1e-8, 
    "rho_max": 1e+16, 
    "w_threshold": 0.3,
    "lambda": 0.1,
    "progress_rate": 0.1,
    "rho_rate": 10.,
}
#%%
# if params['neptune']:
try:
    import neptune.new as neptune
except:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "neptune-client"])
    import neptune.new as neptune

from neptune.new.types import File

with open("../neptune_api.txt", "r") as f:
    key = f.readlines()

run = neptune.init(
    project="an-seunghwan/causal",
    api_token=key[0],
    # run="",
)  

run["sys/name"] = "causal_notears_experiment"
run["sys/tags"].add(["notears", "linear", "torch"])
run["model/params"] = params

# model_version["model/environment"].upload("environment.yml")
#%%
'''simulate DAG and weighted adjacency matrix'''
set_random_seed(params["seed"])
B_true = simulate_dag(params["d"], params["s0"], params["graph_type"])
W_true = simulate_parameter(B_true)

run["model/params/W"].upload(File.as_html(pd.DataFrame(W_true)))
run["pickle/W"].upload(File.as_pickle(pd.DataFrame(W_true)))
fig = viz_graph(W_true, size=(7, 7), show=True)
run["model/params/G"].upload(fig)
#%%
'''simulate dataset'''
X = simulate_linear_sem(W_true, params["n"], params["sem_type"])
run["model/data"].upload(File.as_html(pd.DataFrame(X)))
run["pickle/data"].upload(File.as_pickle(pd.DataFrame(X)))
#%%








#%%
n, d = X.shape

assert n == params["n"]
assert d == params["d"]

w_est = np.zeros(2 * d * d) # double w_est into (w_pos, w_neg)
rho = params["rho"]
alpha = params["alpha"]
h = params["h"]

'''?'''
bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]

X = X - np.mean(X, axis=0, keepdims=True) # normalize
#%%
def _adj(w):
    """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
    return (w[:d * d] - w[d * d:]).reshape([d, d])

def _h(W):
    """Evaluate value and gradient of acyclicity constraint."""
    E = slin.expm(W * W)  # (Zheng et al. 2018)
    h = np.trace(E) - d
    grad_h = E.T * W * 2
    return h, grad_h

def _func(w):
    """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
    W = _adj(w)
    
    R = X - X @ W
    loss = 0.5 / n * (R ** 2).sum() + (params["lambda"] * w.sum())
    
    h, grad_h= _h(W)
    obj = loss + (0.5 * params["rho"] * h * h) + (params["alpha"] * h)
    
    grad = - 1. / n * X.T @ R
    grad += (params["alpha"] + params["rho"] * h) * grad_h
    g_obj = np.concatenate((grad + params["lambda"], -grad + params["lambda"]), axis=None)
    return obj, g_obj
#%%
for iteration in range(params["max_iter"]):
    w_new, h_new = None, None
    while rho < params["rho_max"]:
        sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
        w_new = sol.x
        h_new, _ = _h(_adj(w_new))
        if h_new > params["progress_rate"] * h:
            rho *= params["rho_rate"]
        else:
            break
    # update solution
    w_est, h = w_new, h_new
    # dual ascent step
    alpha += params["rho"] * h
    # stopping rules
    if h <= params["h_tol"] or rho >= params["rho_max"]:
        break
    
    """update log"""
    run["train/iteration/rho"].log(rho)
    run["train/iteration/h"].log(h)
    run["train/iteration/alpha"].log(alpha)
    
W_est = _adj(w_est)
W_est[np.abs(W_est) < params["w_threshold"]] = 0
#%%
"""chech DAGness of estimated weighted graph"""
W_est = np.round(W_est, 2)

fig = plt.figure(figsize=(9, 9))
G = nx.from_numpy_matrix(W_est, create_using=nx.DiGraph)
layout = nx.planar_layout(G)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw(G, layout, 
        with_labels=True, 
        font_size=20,
        font_weight='bold',
        arrowsize=40,
        node_size=1000)
nx.draw_networkx_edge_labels(G, 
                             pos=layout, 
                             edge_labels=labels, 
                             font_weight='bold',
                             font_size=15)
run["result/G_est"].upload(fig)
plt.show()
plt.close()

# assert ig.Graph.Weighted_Adjacency(W_est.tolist()).is_dag()
#%%
run["result/Is DAG?"] = ig.Graph.Weighted_Adjacency(W_est.tolist()).is_dag()
run["result/W_est"].upload(File.as_html(pd.DataFrame(W_est)))
run["pickle/W_est"].upload(File.as_pickle(pd.DataFrame(W_est)))
run["result/W_diff"].upload(File.as_html(pd.DataFrame(W - W_est)))
#%%
W_ = (W != 0).astype(float)
W_est_ = (W_est != 0).astype(float)
W_diff_ = np.abs(W_ - W_est_)

fig = plt.figure(figsize=(9, 9))
G = nx.from_numpy_matrix(W_diff_, create_using=nx.DiGraph)
layout = nx.planar_layout(G)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw(G, layout, 
        with_labels=True, 
        font_size=20,
        font_weight='bold',
        arrowsize=40,
        node_size=1000)
nx.draw_networkx_edge_labels(G, 
                             pos=layout, 
                             edge_labels=labels, 
                             font_weight='bold',
                             font_size=15)
run["result/G_diff"].upload(fig)
plt.show()
plt.close()
#%%
run.stop()
#%%