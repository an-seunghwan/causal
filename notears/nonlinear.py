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
config = {
    "seed": 10,
    "n": 200,
    "d": 5,
    "s0": 5,
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
    "lambda1": 0.001,
    "lambda2": 0.001,
    "progress_rate": 0.25,
    "rho_max": 1e+16, 
    "rho_rate": 10.,
}
#%%
import sys
import subprocess
try:
    import wandb
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    with open("../wandb_api.txt", "r") as f:
        key = f.readlines()
    subprocess.run(["wandb", "login"], input=key[0], encoding='utf-8')
    import wandb

wandb.init(project="causal", 
            entity="anseunghwan",
            tags=["notears", "nonlinear", "torch"])

wandb.config = config
#%%
'''simulate DAG and weighted adjacency matrix'''
set_random_seed(config["seed"])
B_true = simulate_dag(config["d"], config["s0"], config["graph_type"])

wandb.run.summary['B_true'] = wandb.Table(data=pd.DataFrame(B_true))
fig = viz_graph(B_true, size=(7, 7))
wandb.log({'Graph': wandb.Image(fig)})
fig = viz_heatmap(B_true, size=(5, 4))
wandb.log({'heatmap': wandb.Image(fig)})
#%%
'''simulate dataset'''
X = simulate_nonlinear_sem(B_true, 
                            config["n"], 
                            hidden_dims=config["hidden_dims"], 
                            activation='sigmoid', 
                            weight_range=(0.5, 2.), 
                            noise_scale=None)
n, d = X.shape
assert n == config["n"]
assert d == config["d"]
wandb.run.summary['data'] = wandb.Table(data=pd.DataFrame(X))
#%%
def loss_function(model, X, alpha, rho, config):
    """Evaluate value and gradient of augmented Lagrangian."""
    R = X - model(X)
    loss = 0.5 / config["n"] * (R ** 2).sum() + config["lambda1"] * torch.norm(model.W_est, p=1)
    
    h = model.h_fun()
    loss += (0.5 * rho * (h ** 2))
    loss += (alpha * h)
    
    for mlp in model.MLP:
        for dense in mlp:
            loss += 0.5 * config["lambda2"] * torch.norm(dense.weight, p=2)
    loss += 0.5 * config["lambda2"] * torch.norm(model.W_est, p=2)
    return loss
#%%
"""model define"""
class NotearsMLP(nn.Module):
    def __init__(self, config):
        super(NotearsMLP, self).__init__()
        self.config = config
        self.W_est = nn.Parameter(torch.zeros((self.config["d"], self.config["d"])))
        
        in_dim = self.config["d"]
        self.MLP = []
        for j in range(self.config["d"]):
            dense = []
            for out_dim in self.config["hidden_dims"] + [1]:
                if out_dim != 1:
                    dense.append(nn.Linear(in_dim, out_dim))
                else:
                    dense.append(nn.Linear(in_dim, out_dim, bias=False))
                in_dim = out_dim
            self.MLP.append(nn.ModuleList(dense))
            in_dim = self.config["d"]

    def forward(self, x):  # [n, d] -> [n, d]
        w_split = torch.split(self.W_est, 1, dim=1)
        result = []
        for j in range(config["d"]):
            h = x * w_split[j].t()
            for mlp in self.MLP[j]:
                h = mlp(h)
                h = nn.Sigmoid()(h)
            result.append(h)
        return torch.cat(result, dim=1)
    
    def h_fun(self):
        """Evaluate DAGness constraint"""
        h = trace_expm(self.W_est * self.W_est) - self.W_est.shape[0]
        return h

model = NotearsMLP(config)
#%%
"""optimization process"""
# initial values
rho = config["rho"]
alpha = config["alpha"]
h = config["h"]

optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
#%%
for iteration in range(config["max_iter"]):
    # primal update
    count = 0
    h_old = np.inf
    while True:
        optimizer.zero_grad()
        loss = loss_function(model, X, alpha, rho, config)
        loss.backward()
        optimizer.step()
        
        h_new = model.h_fun().item()
        if h_new < config["progress_rate"] * h: 
            break
        elif abs(h_old - h_new) < 1e-8: # no change in weight estimation
            if rho >= config["rho_max"]:
                break
            else:
                rho *= config["rho_rate"]
        h_old = h_new
            
        count += 1
        if count % 10 == 0:
            """update log"""
            wandb.log({"inner_loop/h_new": h_new})
            wandb.log({"inner_loop/loss": loss.item()})
        
    # update
    h = h_new
    # dual ascent step
    alpha += rho * h
    # stopping rules
    if h <= config["h_tol"] or rho >= config["rho_max"]: 
        break
    
    """update log"""
    wandb.log({"train/h": h})
    wandb.log({"train/alpha": alpha})
    wandb.log({"train/loss": loss.item()})
    
    print('[iteration {:03d}]: loss: {:.4f}, h(W): {:.4f}, primal update: {:04d}'.format(
        iteration, loss.item(), h, count))
#%%
"""chech DAGness of estimated weighted graph"""
W_est = model.W_est.detach().numpy().astype(float).round(2)
W_est[np.abs(W_est) < config["w_threshold"]] = 0.
B_est = (W_est != 0).astype(float)

fig = viz_graph(W_est, size=(7, 7))
wandb.log({'Graph_est': wandb.Image(fig)})
fig = viz_heatmap(W_est, size=(5, 4))
wandb.log({'heatmap_est': wandb.Image(fig)})

wandb.run.summary['IsDAG?'] = is_dag(W_est)
wandb.run.summary['W_est'] = wandb.Table(data=pd.DataFrame(W_est))
wandb.run.summary['W_diff'] = wandb.Table(data=pd.DataFrame(B_true - B_est))
fig = viz_graph(np.abs(B_true - B_est), size=(7, 7))
wandb.log({'Graph_diff': wandb.Image(fig)})
#%%
"""accuracy"""
acc = count_accuracy(B_true, B_est)
wandb.run.summary['acc'] = acc
#%%
wandb.run.finish()
#%%