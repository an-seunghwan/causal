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

from utils.model import (
    Encoder,
    Decoder
)
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
    project="(causal)DAG-GNN", 
    entity="anseunghwan",
    tags=["nonlinear"],
)
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results')
    parser.add_argument('--n', default=5000, type=int,
                        help='the number of dataset')
    parser.add_argument('--d', default=10, type=int,
                        help='the number of nodes')
    parser.add_argument('--degree', default=2, type=int,
                        help='expected number of edges')
    parser.add_argument('--graph_type', type=str, default='ER',
                        help='graph type: ER, SF, BP')
    parser.add_argument('--sem_type', type=str, default='gauss',
                        help='sem type: gauss, exp, gumbel, uniform, logistic, poisson')
    parser.add_argument('--nonlinear_type', type=str, default='nonlinear_2',
                        help='nonlinear causal structure type: nonlinear_1, nonlinear_2')

    parser.add_argument('--rho', default=1, type=float,
                        help='rho')
    parser.add_argument('--alpha', default=0, type=float,
                        help='alpha')
    parser.add_argument('--h', default=np.inf, type=float,
                        help='h')
    
    parser.add_argument("--hidden", default=64, type=int,
                        help="hidden dimensions for MLP")
    parser.add_argument("--num_layer", default=2, type=int,
                        help="hidden dimensions for MLP")
    parser.add_argument("--x_dim", default=1, type=int,
                        help="dimension of each node")
    
    parser.add_argument('--epochs', default=300, type=int,
                        help='maximum iteration')
    parser.add_argument('--batch_size', default=100, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=0.003, type=float,
                        help='learning rate')
    parser.add_argument('--lr_decay', default=200, type=float,
                        help='learning rate decay')
    parser.add_argument('--gamma', default=1, type=float,
                        help='learning rate decay rate')
    
    parser.add_argument('--max_iter', default=100, type=int,
                        help='maximum number of iteration')
    parser.add_argument('--h_tol', default=1e-8, type=float,
                        help='h value tolerance')
    parser.add_argument('--w_threshold', default=0.3, type=float,
                        help='threshold for weighted adjacency matrix')
    parser.add_argument('--lambda', default=0, type=float,
                        help='coefficient of LASSO penalty')
    parser.add_argument('--progress_rate', default=0.25, type=float,
                        help='progress rate')
    parser.add_argument('--rho_max', default=1e+20, type=float,
                        help='maximum rho value')
    parser.add_argument('--rho_rate', default=10, type=float,
                        help='rho rate')
    
    parser.add_argument('--fig_show', default=False, type=bool)

    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
    
# config = {
#     "seed": 1,
#     'data_type': 'synthetic', # discrete, real
#     "n": 5000,
#     "d": 10,
#     "degree": 2,
#     "x_dim": 1,
#     "graph_type": "ER",
#     "sem_type": "gauss",
#     "nonlinear_type": "nonlinear_2",
#     "hidden": 64,
    
#     "epochs": 300,
#     "lr": 0.003,
#     "lr_decay": 200,
#     "gamma": 1.,
#     "batch_size": 100,
    
#     "rho": 1, # initial value
#     "alpha": 0., # initial value
#     "h": np.inf, # initial value
    
#     "max_iter": 100, 
#     "loss_type": 'l2',
#     "h_tol": 1e-8, 
#     "w_threshold": 0.3,
#     "lambda": 0.,
#     "progress_rate": 0.25,
#     "rho_max": 1e+20, 
#     "rho_rate": 10,
    
#     "fig_show": False,
# }
#%%
def h_fun(A, d):
    x = torch.eye(d).float() + torch.div(A * A, d) # alpha = 1 / d
    return torch.trace(torch.matrix_power(x, d)) - d
#%%
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
def train(train_loader, encoder, decoder, rho, alpha, config, optimizer):
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
        recon = 0.5 * torch.pow(recon - train_batch, 2).sum() / train_batch.size(0)
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
def main():
    config = vars(get_args(debug=False)) # default configuration
    config["cuda"] = torch.cuda.is_available()
    wandb.config.update(config)

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
    
    """initialize adjacency matrix A"""
    adj_A = np.zeros((config["d"], config["d"]))
    
    encoder = Encoder(config, adj_A, config["hidden"])
    decoder = Decoder(config, config["hidden"])

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

    rho = config["rho"]
    alpha = config["alpha"]
    h = config["h"]
        
    for iteration in range(config["max_iter"]):
        
        """primal problem"""
        while rho < config["rho_max"]:
            # find argmin of primal problem (local solution) = update for config["epochs"] times
            for epoch in tqdm.tqdm(range(config["epochs"]), desc="primal update"):
                logs, adj_A_amplified = train(train_loader, encoder, decoder, rho, alpha, config, optimizer)
            # only one epoch is fine for finding argmin
            # logs, adj_A_amplified = train(rho, alpha, config, optimizer)
            
            W_est = adj_A_amplified.data.clone()
            h_new = h_fun(W_est, config["d"])
            if h_new.item() > config["progress_rate"] * h:
                rho *= config["rho_rate"]
            else:
                break
        
        """dual ascent"""
        h = h_new.item()
        alpha += rho * h_new.item()
        
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

    """accuracy"""
    acc = count_accuracy(B_true, B_est)
    wandb.log(acc)
    
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%