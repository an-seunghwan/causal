#%%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import numpy as np
np.set_printoptions(precision=3)
import pandas as pd

import torch
import torch.nn as nn
torch.set_default_dtype(torch.float)

from utils.simulation import (
    is_dag,
    set_random_seed,
    simulate_dag,
    simulate_nonlinear_sem,
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
    "n": 200,
    "d": 5,
    "s0": 9,
    "graph_type": 'ER',
    "sem_type": 'mlp', # only mlp
    "hidden_dims": [16, 32],
    
    "rho": 1, # initial value
    "alpha": 0., # initial value
    "h": np.inf, # initial value
    
    "lr": 0.001,
    "loss_type": 'l2',
    "max_iter": 100, 
    "h_tol": 1e-8, 
    "w_threshold": 0.1,
    "lambda1": 0.01,
    "lambda2": 0.05,
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
run["sys/tags"].add(["notears", "nonlinear", "torch"])
run["model/params"] = params

# model_version["model/environment"].upload("environment.yml")
#%%
'''simulate DAG and weighted adjacency matrix'''
set_random_seed(params["seed"])
B_true = simulate_dag(params["d"], params["s0"], params["graph_type"])

run["model/params/B"].upload(File.as_html(pd.DataFrame(B_true)))
run["pickle/B"].upload(File.as_pickle(pd.DataFrame(B_true)))
fig = viz_graph(B_true, size=(7, 7), show=True)
run["model/params/Graph"].upload(fig)
fig = viz_heatmap(B_true, size=(5, 4), show=True)
run["model/params/heatmap"].upload(fig)
#%%
'''simulate dataset'''
X = simulate_nonlinear_sem(B_true, 
                            params["n"], 
                            hidden_dims=params["hidden_dims"], 
                            activation='sigmoid', 
                            weight_range=(0.5, 2.), 
                            noise_scale=None)
n, d = X.shape
assert n == params["n"]
assert d == params["d"]
# run["model/data"].upload(File.as_html(pd.DataFrame(X)))
run["pickle/data"].upload(File.as_pickle(pd.DataFrame(X)))
#%%
def loss_function(model, X, alpha, rho, params):
    """Evaluate value and gradient of augmented Lagrangian."""
    R = X - model(X)
    loss = 0.5 / params["n"] * (R ** 2).sum() + params["lambda1"] * torch.norm(model.W_est, p=1)
    
    h = model.h_fun()
    loss += (0.5 * rho * (h ** 2))
    loss += (alpha * h)
    
    for mlp in model.MLP:
        loss += 0.5 * params["lambda2"] * (mlp.weight ** 2).sum()
    return loss
#%%
"""model define"""
class NotearsMLP(nn.Module):
    def __init__(self, params):
        super(NotearsMLP, self).__init__()
        self.params = params
        self.W_est = nn.Parameter(torch.zeros((self.params["d"], self.params["d"])))
        in_dim = self.params["d"]
        dense = []
        for out_dim in self.params["hidden_dims"] + [1]:
            if out_dim != 1:
                dense.append(nn.Linear(in_dim, out_dim))
            else:
                dense.append(nn.Linear(in_dim, out_dim, bias=False))
            in_dim = out_dim
        self.MLP = nn.ModuleList(dense)

    def forward(self, x):  # [n, d] -> [n, d]
        x = x.repeat((1, self.params["d"])) * self.W_est.view(1, -1)
        x = x.view(self.params["n"] * self.params["d"], self.params["d"])
        for mlp in self.MLP:
            x = mlp(x)
            x = nn.Sigmoid()(x)
        x = x.view(self.params["n"], self.params["d"])
        return x
    
    def h_fun(self):
        """Evaluate DAGness constraint"""
        h = trace_expm(self.W_est * self.W_est) - self.W_est.shape[0]
        return h

model = NotearsMLP(params)
#%%
"""optimization process"""
# initial values
rho = params["rho"]
alpha = params["alpha"]
h = params["h"]

optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
#%%
for iteration in range(params["max_iter"]):
    # primal update
    count = 0
    h_old = np.inf
    while True:
        optimizer.zero_grad()
        loss = loss_function(model, X, alpha, rho, params)
        loss.backward()
        optimizer.step()
        
        h_new = model.h_fun().item()
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
W_est = model.W_est.detach().numpy().astype(float).round(2)
W_est[np.abs(W_est) < params["w_threshold"]] = 0.
B_est = (W_est != 0).astype(float)

fig = viz_graph(W_est, size=(7, 7), show=True)
run["result/Graph_est"].upload(fig)
fig = viz_heatmap(W_est, size=(5, 4), show=True)
run["result/heatmap_est"].upload(fig)

run["result/Is DAG?"] = is_dag(W_est)
run["result/W_est"].upload(File.as_html(pd.DataFrame(W_est)))
run["pickle/W_est"].upload(File.as_pickle(pd.DataFrame(W_est)))
run["result/W_diff"].upload(File.as_html(pd.DataFrame(B_true - B_est)))
fig = viz_graph(np.abs(B_true - B_est), size=(7, 7), show=True)
run["result/Graph_diff"].upload(fig)
#%%
"""accuracy"""
acc = count_accuracy(B_true, B_est)
run["result/accuracy"] = acc
#%%
run.stop()
#%%