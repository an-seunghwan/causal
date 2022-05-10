#%%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm

import igraph as ig

import torch
import scipy.linalg as slin

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
    
    "lr": 0.001,
    "loss_type": 'l2',
    "max_iter": 100, 
    "h_tol": 1e-8, 
    "w_threshold": 0.3,
    "lambda": 0.01,
    "progress_rate": 0.9,
    # "rho_max": 1e+16, 
    # "rho_rate": 10.,
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
n, d = X.shape
assert n == params["n"]
assert d == params["d"]
run["model/data"].upload(File.as_html(pd.DataFrame(X)))
run["pickle/data"].upload(File.as_pickle(pd.DataFrame(X)))
#%%
'''?????'''
class TraceExpm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # detach so we can cast to NumPy
        E = slin.expm(input.detach().numpy())
        f = np.trace(E)
        E = torch.from_numpy(E)
        ctx.save_for_backward(E)
        return torch.as_tensor(f, dtype=input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        E, = ctx.saved_tensors
        grad_input = grad_output * E.t()
        return grad_input

trace_expm = TraceExpm.apply
#%%
def loss_function(X, W_est, alpha, rho):
    """Evaluate value and gradient of augmented Lagrangian."""
    R = X - X.matmul(W_est)
    loss = 0.5 * torch.pow(R, 2).mean() + params["lambda"] * W_est.abs().sum()
    
    h = trace_expm(W_est) - W_est.shape[0]
    loss += (0.5 * rho * torch.pow(h, 2)) + (alpha * h)
    return loss
#%%
'''optimization process'''
X = X - np.mean(X, axis=0, keepdims=True) # normalize
X = torch.FloatTensor(X)

W_est = torch.zeros((params["d"], params["d"]), 
                    requires_grad=True)

# initial values
rho = params["rho"]
alpha = params["alpha"]
# h = float((trace_expm(W_est * W_est) - params["d"]).item())
h = params["h"]

optimizer = torch.optim.SGD([W_est], lr=params["lr"])
#%%
for iteration in tqdm.tqdm(range(params["max_iter"])):
    # primal update
    while True:
        optimizer.zero_grad()
        loss = loss_function(X, W_est, alpha, rho)
        loss.backward()
        optimizer.step()
        
        h_new = trace_expm(W_est) - params["d"]
        h_new = h_new.item()
        
        if h_new < params["progress_rate"] * h: break
        
    # update
    h = float((trace_expm(W_est * W_est) - params["d"]).item())
    # dual ascent step
    alpha += rho * h
    # stopping rules
    if h <= params["h_tol"]: break
   
    """update log"""
    run["train/iteration/h"].log(h)
    run["train/iteration/alpha"].log(alpha)
    # run["train/iteration/rho"].log(rho)
    
    print(loss.item(), h_new, rho)
    
W_est = W_est.detach().numpy().round(2)
W_est[np.abs(W_est) < params["w_threshold"]] = 0
#%%
"""chech DAGness of estimated weighted graph"""
fig = viz_graph(W_est, size=(7, 7), show=True)
run["result/G_est"].upload(fig)
# assert ig.Graph.Weighted_Adjacency(W_est.tolist()).is_dag()
#%%
run["result/Is DAG?"] = ig.Graph.Weighted_Adjacency(W_est.tolist()).is_dag()
run["result/W_est"].upload(File.as_html(pd.DataFrame(W_est)))
run["pickle/W_est"].upload(File.as_pickle(pd.DataFrame(W_est)))
run["result/W_diff"].upload(File.as_html(pd.DataFrame(W_true - W_est)))
#%%
W_ = (W_true != 0).astype(float)
W_est_ = (W_est != 0).astype(float)
W_diff_ = np.abs(W_ - W_est_)

fig = viz_graph(W_diff_, size=(7, 7), show=True)
run["result/G_diff"].upload(fig)
#%%
run.stop()
#%%