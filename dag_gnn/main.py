#%%
import enum
import os

from notears.utils.simulation import count_accuracy
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import numpy as np
import pandas as pd

import torch

from utils.simulation import (
    set_random_seed,
    is_dag,
    load_data,
    count_accuracy
)

from utils.viz import (
    viz_graph,
    viz_heatmap,
)

from utils.utils import (
    encode_onehot
)

from utils.model import (
    Encoder,
    Decoder
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
torch.manual_seed(config["seed"])
train_loader, W_true = load_data(config)

wandb.run.summary['W_true'] = wandb.Table(data=pd.DataFrame(W_true))
fig = viz_graph(W_true, size=(7, 7), show=True)
wandb.log({'Graph': wandb.Image(fig)})
fig = viz_heatmap(W_true, size=(5, 4))
wandb.log({'heatmap': wandb.Image(fig)})
#%%
# """Generate off-diagonal interaction graph"""
# off_diagonal = np.ones((config["d"], config["d"])) - np.eye(config["d"])
# rel_rec = torch.DoubleTensor(np.array(encode_onehot(np.where(off_diagonal)[1]), dtype=np.float64))
# rel_send = torch.DoubleTensor(np.array(encode_onehot(np.where(off_diagonal)[0]), dtype=np.float64))

"""initialize adjacency matrix A"""
adj_A = np.zeros((config["d"], config["d"]))
#%%
def h_fun(A, d):
    x = torch.eye(d).float() + torch.div(A * A, d) # alpha = 1 / d
    return torch.trace(torch.matrix_power(x, d)) - d
#%%
for batch_num, (train_batch, relations) in enumerate(train_loader):
    if batch_num == 0: break
train_batch.shape
#%%
encoder = Encoder(config, adj_A, 32)
decoder = Decoder(config, 32)
#%%
logits, h, adj_A_amplified = encoder(train_batch)
recon, z = decoder(logits, adj_A_amplified, encoder.Wa)
#%%
rho = config["rho"]
alpha = config["alpha"]
h = config["h"]

# reconstruction
recon_loss = torch.pow(recon - train_batch, 2).sum() / train_batch.size(0)

# KL-divergence
kl = torch.pow(logits, 2).sum()
kl = 0.5 * kl / logits.size(0)

# elbo
elbo = recon_loss + kl

L1_reg = config["lambda"] * torch.sum(torch.abs(adj_A_amplified))
loss = elbo + L1_reg # sparsity loss

h_A = h_fun(adj_A_amplified, config["d"])
loss += 0.5 * rho * (h_A ** 2)
loss += alpha * h_A
loss += 100. * torch.trace(adj_A_amplified * adj_A_amplified)
#%%
if torch.sum(adj_A_amplified != adj_A_amplified):
    print('nan error\n')
#%%
# compute metrics
W_est = adj_A_amplified.data.clone().numpy()
acc = count_accuracy(W_true, W_est)
#%%
wandb.run.finish()
#%%