#%%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import numpy as np
import pandas as pd
import tqdm

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

from utils.model import (
    GAE
)

from utils.trac_exp import trace_expm
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
    project="(causal)GAE", 
    entity="anseunghwan",
    tags=["nonlinear"],
)
#%%
config = {
    "seed": 1,
    'data_type': 'synthetic', # discrete, real
    "n": 5000,
    "d": 10,
    "degree": 2,
    "graph_type": "ER",
    "sem_type": "gauss",
    "nonlinear_type": "nonlinear_1",
    "hidden_dim": 16,
    "num_layer": 2,
    "x_dim": 1,
    "latent_dim": 1,
    
    "epochs": 300,
    "batch_size": 100,
    "lr": 0.003,
    "init_iter": 5,
    "early_stopping": True,
    "early_stopping_threshold": 1.15,
    
    "rho": 1, # initial value
    "alpha": 0., # initial value
    "h": np.inf, # initial value
    
    "max_iter": 20, 
    "loss_type": 'l2',
    "h_tol": 1e-12, 
    "w_threshold": 0.2,
    "lambda": 0.,
    "progress_rate": 0.25,
    "rho_max": 1e+18, 
    "rho_rate": 10,
    
    "fig_show": False,
}
#%%
config["cuda"] = torch.cuda.is_available()

set_random_seed(config["seed"])
torch.manual_seed(config["seed"])
if config["cuda"]:
    torch.cuda.manual_seed(config["seed"])
train_loader, W_true = load_data(config)

wandb.run.summary['W_true'] = wandb.Table(data=pd.DataFrame(W_true))
fig = viz_graph(W_true, size=(7, 7), show=config["fig_show"])
wandb.log({'Graph': wandb.Image(fig)})
fig = viz_heatmap(W_true, size=(5, 4), show=config["fig_show"])
wandb.log({'heatmap': wandb.Image(fig)})
#%%
def h_fun(W):
    """Evaluate DAGness constraint"""
    h = trace_expm(W * W) - W.shape[0]
    return h
#%%
model = GAE(config)

if config["cuda"]:
    model.cuda()

optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=config["lr"]
)
#%%
def train(model, rho, alpha, config, optimizer):
    model.train()
    
    if config["cuda"]:
        X = X.cuda()
    
    logs = {
        'loss': [], 
        'recon': [],
        'L1': [],
        'aug': [],
    }
    
    for batch_num, [train_batch] in enumerate(train_loader):
        if config["cuda"]:
            train_batch = train_batch.cuda()
            
        optimizer.zero_grad()
        
        recon = model(train_batch)
        
        loss_ = []    
        
        # reconstruction
        recon = 0.5 * torch.pow(recon - train_batch, 2).sum() / train_batch.size(0)
        loss_.append(('recon', recon))

        # sparsity loss
        L1 = config["lambda"] * torch.sum(torch.abs(model.W))
        loss_.append(('L1', L1))

        # augmentation and lagrangian loss
        h_A = h_fun(model.W)
        aug = 0.5 * rho * (h_A ** 2)
        aug += alpha * h_A
        loss_.append(('aug', aug))
        
        loss = sum([y for _, y in loss_])
        loss_.append(('loss', loss))
        
        loss.backward()
        optimizer.step()
        
        """accumulate losses"""
        for x, y in loss_:
            logs[x] = logs.get(x) + [y.item()]
    
    return logs
#%%
rho = config["rho"]
alpha = config["alpha"]
h = config["h"]
mse_save = np.inf
W_save = None
    
for iteration in range(config["max_iter"]):
    
    """primal problem"""
    while rho < config["rho_max"]:
        # h_old = np.inf
        # find argmin of primal problem (local solution) = update for config["epochs"] times
        for epoch in tqdm.tqdm(range(config["epochs"]), desc="primal update"):
            logs = train(model, rho, alpha, config, optimizer)
            
            # """FIXME"""
            # W_est = model.W.detach().data.clone()
            # h_new = h_fun(W_est).item()
            # # no change in weight estimation (convergence)
            # if abs(h_old - h_new) < 1e-12: 
            #     break
            # h_old = h_new
            
        # only one epoch is fine for finding argmin
        # logs = train(model, X, rho, alpha, config, optimizer)
        
        W_est = model.W.detach().data.clone()
        h_current = h_fun(W_est)
        if h_current.item() > config["progress_rate"] * h:
            rho *= config["rho_rate"]
        else:
            break
    
    if config["early_stopping"]:
        if np.mean(logs['recon']) / mse_save > config["early_stopping_threshold"] and h_current.item() <= 1e-7:
            W_est = W_save
            print("early stopping!")
            break
        else:
            W_save = W_est
            mse_save = np.mean(logs['recon'])
    
    """dual ascent"""
    h = h_current.item()
    alpha += rho * h_current.item()
    
    print_input = "[iteration {:03d}]".format(iteration)
    print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y).round(2)) for x, y in logs.items()])
    print_input += ', h(W): {:.8f}'.format(h)
    print(print_input)
    
    """update log"""
    wandb.log({x : np.mean(y) for x, y in logs.items()})
    wandb.log({'h(W)' : h})
    
    """stopping rule"""
    if h_current.item() <= config["h_tol"] and iteration > config["init_iter"]:
        break
#%%
"""final metrics"""
W_est = W_est.numpy()
W_est = W_est / np.max(np.abs(W_est)) # normalize weighted adjacency matrix
W_est[np.abs(W_est) < config["w_threshold"]] = 0.
W_est = W_est.astype(float).round(2)

fig = viz_graph(W_est, size=(7, 7), show=config["fig_show"])
wandb.log({'Graph_est': wandb.Image(fig)})
fig = viz_heatmap(W_est, size=(5, 4), show=config["fig_show"])
wandb.log({'heatmap_est': wandb.Image(fig)})

wandb.run.summary['Is DAG?'] = is_dag(W_est)
wandb.run.summary['W_est'] = wandb.Table(data=pd.DataFrame(W_est))
wandb.run.summary['W_diff'] = wandb.Table(data=pd.DataFrame(W_true - W_est))

W_ = (W_true != 0).astype(float)
W_est_ = (W_est != 0).astype(float)
W_diff_ = np.abs(W_ - W_est_)

fig = viz_graph(W_diff_, size=(7, 7))
wandb.log({'Graph_diff': wandb.Image(fig)})

B_est = (W_est != 0).astype(float)
B_true = (W_true != 0).astype(float)

# compute metrics
acc = count_accuracy(B_true, B_est)
wandb.log(acc)
#%%
wandb.config.update(config)
wandb.run.finish()
#%%