#%%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import numpy as np
import pandas as pd

import torch

from utils.simulation import (
    set_random_seed,
    is_dag,
    load_data,
)

from utils.viz import (
    viz_graph,
    viz_heatmap,
)
#%%
config = {
    "seed": 1,
    'data_type': 'synthetic', # discrete, real
    "n": 5000,
    "d": 5,
    "degree": 3,
    "x_dim": 3,
    "graph_type": "ER",
    "sem_type": "gauss",
    "nonlinear_type": "nonlinear_1",
    
    "lr": 0.001,
    "batch_size": 128,
    
    "rho": 1, # initial value
    "alpha": 0., # initial value
    "h": np.inf, # initial value
    
    "loss_type": 'l2',
    "max_iter": 100, 
    "h_tol": 1e-8, 
    "w_threshold": 0.3,
    "lambda": 0.005,
    "progress_rate": 0.25,
    "rho_max": 1e+16, 
    "rho_rate": 2.,
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

wandb.init(
    project="causal", 
    entity="anseunghwan",
    tags=["DAG-GNN", "nonlinear"],
    # name='notears'
)
#%%
set_random_seed(config["seed"])
train_loader, W_true = load_data(config)

wandb.run.summary['W_true'] = wandb.Table(data=pd.DataFrame(W_true))
fig = viz_graph(W_true, size=(7, 7), show=True)
wandb.log({'Graph': wandb.Image(fig)})
fig = viz_heatmap(W_true, size=(5, 4))
wandb.log({'heatmap': wandb.Image(fig)})
#%%

#%%
wandb.run.finish()
#%%