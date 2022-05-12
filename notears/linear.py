#%%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import numpy as np
import pandas as pd

import torch

from utils.simulation import (
    is_dag,
    set_random_seed,
    simulate_dag,
    simulate_parameter,
    simulate_linear_sem,
    count_accuracy,
)

from utils.viz import (
    viz_graph,
    viz_heatmap,
)

from utils.trac_exp import trace_expm
#%%
params = {
    # "neptune": True, # True if you use neptune.ai
    
    "seed": 10,
    "n": 1000,
    "d": 7,
    "s0": 7,
    "graph_type": 'ER',
    "sem_type": 'gauss',
    
    "rho": 1, # initial value
    "alpha": 0., # initial value
    "h": np.inf, # initial value
    
    "lr": 0.001,
    "loss_type": 'l2',
    "max_iter": 100, 
    "h_tol": 1e-8, 
    "w_threshold": 0.2,
    "lambda": 0.005,
    "progress_rate": 0.25,
    "rho_max": 1e+16, 
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
run["model/params/Graph"].upload(fig)
fig = viz_heatmap(W_true, size=(5, 4), show=True)
run["model/params/heatmap"].upload(fig)
#%%
'''simulate dataset'''
X = simulate_linear_sem(W_true, params["n"], params["sem_type"], normalize=True)
n, d = X.shape
assert n == params["n"]
assert d == params["d"]
run["model/data"].upload(File.as_html(pd.DataFrame(X)))
run["pickle/data"].upload(File.as_pickle(pd.DataFrame(X)))
#%%
def h_fun(W):
    """Evaluate DAGness constraint"""
    h = trace_expm(W * W) - W.shape[0]
    return h
#%%
def loss_function(X, W_est, alpha, rho):
    """Evaluate value and gradient of augmented Lagrangian."""
    R = X - X.matmul(W_est)
    # loss = 0.5 / params["n"] * (R ** 2).sum() + params["lambda"] * W_est.abs().sum()
    loss = 0.5 / params["n"] * (R ** 2).sum() + params["lambda"] * torch.norm(W_est, p=1)
    
    h = h_fun(W_est)
    loss += (0.5 * rho * (h ** 2))
    loss += (alpha * h)
    return loss
#%%
'''optimization process'''
X = torch.FloatTensor(X)

W_est = torch.zeros((params["d"], params["d"]), 
                    requires_grad=True)

# initial values
rho = params["rho"]
alpha = params["alpha"]
h = params["h"]

optimizer = torch.optim.Adam([W_est], lr=params["lr"])
#%%
for iteration in range(params["max_iter"]):
    # primal update
    count = 0
    h_old = np.inf
    while True:
        optimizer.zero_grad()
        loss = loss_function(X, W_est, alpha, rho)
        loss.backward()
        optimizer.step()
        
        h_new = h_fun(W_est).item()
        if h_new < params["progress_rate"] * h: 
            break
        elif abs(h_old - h_new) < 1e-8: # no change in weight estimation
            if rho >= params["rho_max"]:
                break
            else:
                rho *= params["rho_rate"]
        h_old = h_new
            
        count += 1
        if count % 10 == 0:
            """update log"""
            run["train/inner_loop/h_new"].log(h_new)
            run["train/inner_loop/loss"].log(loss.item())
        
    # update
    h = h_new
    # dual ascent step
    alpha += rho * h
    # stopping rules
    if h <= params["h_tol"] or rho >= params["rho_max"]: 
        break
   
    """update log"""
    run["train/iteration/h"].log(h)
    run["train/iteration/alpha"].log(alpha)
    run["train/iteration/loss"].log(loss.item())
    # run["train/iteration/rho"].log(rho)
    
    print('[iteration {:03d}]: loss: {:.4f}, h(W): {:.4f}, primal update: {:04d}'.format(
        iteration, loss.item(), h, count))
#%%
"""chech DAGness of estimated weighted graph"""
W_est = W_est.detach().numpy().astype(float).round(2)
W_est[np.abs(W_est) < params["w_threshold"]] = 0.

fig = viz_graph(W_est, size=(7, 7), show=True)
run["result/Graph_est"].upload(fig)
fig = viz_heatmap(W_est, size=(5, 4), show=True)
run["result/heatmap_est"].upload(fig)

run["result/Is DAG?"] = is_dag(W_est)
run["result/W_est"].upload(File.as_html(pd.DataFrame(W_est)))
run["pickle/W_est"].upload(File.as_pickle(pd.DataFrame(W_est)))
run["result/W_diff"].upload(File.as_html(pd.DataFrame(W_true - W_est)))

W_ = (W_true != 0).astype(float)
W_est_ = (W_est != 0).astype(float)
W_diff_ = np.abs(W_ - W_est_)

fig = viz_graph(W_diff_, size=(7, 7), show=True)
run["result/Graph_diff"].upload(fig)
#%%
"""accuracy"""
B_est = (W_est != 0).astype(float)
acc = count_accuracy(B_true, B_est)
run["result/accuracy"] = acc
#%%
run.stop()
#%%