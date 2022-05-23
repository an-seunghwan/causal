#%%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import numpy as np
import pandas as pd
import math
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

# from utils.model import (
    
# )
#%%
config = {
    "seed": 42,
    'data_type': 'synthetic', # discrete, real
    "n": 3000,
    "d": 20,
    "degree": 3,
    "graph_type": "ER",
    "sem_type": "gauss",
    "nonlinear_type": "nonlinear_2",
    "hidden_dim": 16,
    "num_layer": 2,
    "x_dim": 5,
    "latent_dim": 3,
    
    "epochs": 300,
    "init_epoch": 5,
    "lr": 0.003,
    "lr_decay": 200,
    "gamma": 1.,
    "batch_size": 100,
    "early_stopping": True,
    "early_stopping_threshold": 1.15,
    
    "rho": 1, # initial value
    "alpha": 0., # initial value
    "h": np.inf, # initial value
    
    "max_iter": 20, 
    "loss_type": 'l2',
    "h_tol": 1e-8, 
    "w_threshold": 0.2,
    "lambda": 1.,
    "progress_rate": 0.25,
    "rho_max": 1e+18, 
    "rho_rate": 10,
    
    "fig_show": True,
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
    project="(causal)GAE", 
    entity="anseunghwan",
    tags=["nonlinear"],
)
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
def train(rho, alpha, h, config, optimizer):
    encoder.train()
    decoder.train()
    
    # update optimizer
    optimizer, lr = update_optimizer(optimizer, config["lr"], rho)
    
    logs = {
        'loss': [], 
        'recon': [],
        'kl': [],
        'L1': [],
        'aug': [],
    }
    
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
            
        loss_ = []    
        
        # reconstruction
        recon = torch.pow(recon - train_batch, 2).sum() / train_batch.size(0)
        loss_.append(('recon', recon))

        # KL-divergence
        kl = torch.pow(logits, 2).sum()
        kl = 0.5 * kl / logits.size(0)
        loss_.append(('kl', kl))

        # sparsity loss
        L1 = config["lambda"] * torch.sum(torch.abs(adj_A_amplified))
        loss_.append(('L1', L1))

        # augmentation and lagrangian loss
        h_A = h_fun(adj_A_amplified, config["d"])
        aug = 0.5 * rho * (h_A ** 2)
        aug += alpha * h_A
        """?????"""
        aug += 100. * torch.trace(adj_A_amplified * adj_A_amplified)
        loss_.append(('aug', aug))
        
        loss = sum([y for _, y in loss_])
        loss_.append(('loss', loss))
        
        loss.backward()
        optimizer.step()
        
        """accumulate losses"""
        for x, y in loss_:
            logs[x] = logs.get(x) + [y.item()]
            
    return logs, adj_A_amplified
#%%
rho = config["rho"]
alpha = config["alpha"]
h = config["h"]
    
for iteration in range(config["max_iter"]):
    
    """primal problem"""
    while rho < config["rho_max"]:
        # find argmin of primal problem (local solution) = update for config["epochs"] times
        for epoch in tqdm.tqdm(range(config["epochs"]), desc="primal update"):
            logs, adj_A_amplified = train(rho, alpha, h, config, optimizer)
        # only one epoch is fine for finding argmin
        # logs, adj_A_amplified = train(rho, alpha, h, config, optimizer)
        
        W_est = adj_A_amplified.data.clone()
        h_new = h_fun(W_est, config["d"])
        if h_new.item() > config["progress_rate"] * h:
            rho *= config["rho_rate"]
        else:
            break
    
    """dual ascent"""
    h = h_new.item()
    alpha += config["rho"] * h_new.item()
    
    print_input = "[iteration {:03d}]".format(iteration)
    print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y).round(2)) for x, y in logs.items()])
    print_input += ', h(W): {:.8f}'.format(h)
    print(print_input)
    
    """update log"""
    wandb.log({x : np.mean(y) for x, y in logs.items()})
    wandb.log({'h(W)' : h})
    
    """stopping rule"""
    if h_new.item() <= config["h_tol"]:
        break
#%%
"""final metrics"""
adj_A_amplified = encoder.amplified_adjacency_matrix()
W_est = adj_A_amplified.data.clone().numpy()
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