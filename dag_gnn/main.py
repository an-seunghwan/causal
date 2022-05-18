#%%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import numpy as np
import pandas as pd
import math

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
    
    "hidden": 32,
    
    "epochs": 300,
    "lr": 0.001,
    "lr_decay": 200,
    "gamma": 1.,
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
config["cuda"] = torch.cuda.is_available()

set_random_seed(config["seed"])
torch.manual_seed(config["seed"])
if config["cuda"]:
    torch.cuda.manual_seed(config["seed"])
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
encoder = Encoder(config, adj_A, 32)
decoder = Decoder(config, 32)

if config["cuda"]:
    encoder.cuda()
    decoder.cuda()

optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()), 
    lr=config["lr"]
)
torch.optim.lr_scheduler.StepLR(
    optimizer, 
    step_size=config["lr_decay"],
    gamma=config["gamma"]
)

def update_optimizer(optimizer, lr, rho):
    """related to lr to rho, whenever rho gets big, reduce lr propotionally"""
    MAX_LR = 1e-2
    MIN_LR = 1e-4
    
    estimated_lr = lr / (math.log10(rho) + 1e-10)
    if estimated_lr > MAX_LR:
        lr = MAX_LR
    elif estimated_lr < MIN_LR:
        lr = MIN_LR
    else:
        lr = estimated_lr
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return optimizer, lr
#%%
def train(rho, alpha, h, config):
    encoder.train()
    decoder.train()
    
    # update optimizer
    optimizer, lr = update_optimizer(optimizer, config["lr"], rho)
    
    for batch_num, [train_batch] in enumerate(train_loader):
        if config["cuda"]:
            train_batch = train_batch.cuda()
        
        optimizer.zero_grad()
        
        logits, h, adj_A_amplified = encoder(train_batch)
        recon, z = decoder(logits, adj_A_amplified, encoder.Wa)
        
        if torch.sum(adj_A_amplified != adj_A_amplified):
            print('nan error\n')
        if torch.sum(recon != recon):
            print('nan error\n')
            
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
        
        loss.backward()
        optimizer.step()
        
        W_est = adj_A_amplified.data.clone()
        h_new = h_fun(W_est, config["d"])
        if h_new.item() > config["progress_rate"] * h_old:
            rho *= config["rho_rate"]
        else:
            break

        # W_est = adj_A_amplified.data.clone().numpy()
        # W_est[np.abs(W_est) < config["w_threshold"]] = 0.
        # B_est = (W_est != 0).astype(float)
        # B_true = (W_true != 0).astype(float)

        # # compute metrics
        # acc = count_accuracy(B_true, B_est)
        # wandb.log(acc)
#%%
rho = config["rho"]
alpha = config["alpha"]
h = config["h"]

h_old = h_new.item()
alpha += config["rho"] * h_new.item()

if h_new.item() <= config["h_tol"]:
    break
#%%
wandb.run.finish()
#%%