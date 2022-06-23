#%%
import os

from nocurl.utils.util import build_dag
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import numpy as np
import pandas as pd
import tqdm

import torch

from utils.simulation import (
    set_random_seed,
    is_dag,
    SyntheticDataset,
    count_accuracy
)

from utils.viz import (
    viz_graph,
    viz_heatmap,
)
from utils.util import (
    build_dag
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
    project="(causal)NOCURL", 
    entity="anseunghwan",
    tags=["linear", "golem"],
)
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results')
    parser.add_argument('--n', default=1000, type=int,
                        help='the number of dataset')
    parser.add_argument('--d', default=20, type=int,
                        help='the number of nodes')
    parser.add_argument('--degree', default=4, type=int,
                        help='degree of graph')
    parser.add_argument('--graph_type', type=str, default='ER',
                        help='graph type: ER or SF')
    parser.add_argument('--noise_type', type=str, default='gaussian_ev',
                        help='noise type: gaussian_ev, gaussian_nv, exponential, gumbel')
    parser.add_argument('--B_scale', type=float, default=1,
                        help='scaling factor for range of B')

    parser.add_argument('--lambda', default=2e-2, type=float,
                        help='coefficient of LASSO penalty')
    parser.add_argument('--equal_variances', default=True, type=bool,
                        help="Assume equal noise variances for likelibood objective.")
    # parser.add_argument('--non_equal_variances', default=False, type=bool,
    #                     help="Assume non-equal noise variances for likelibood objective.")
    
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate')
    parser.add_argument('--epochs', default=10000, type=int,
                        help='number of iterations for training')
    
    parser.add_argument('--w_threshold', default=0.3, type=float,
                        help='Threshold used to determine whether has edge in graph, element greater'
                            'than the w_threshold means has a directed edge, otherwise has not.')
    
    parser.add_argument('--fig_show', default=True, type=bool)

    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def h_function(w, config):
    h_A = torch.trace(torch.matrix_exp(w * w)) - config["d"]
    return h_A
#%%
def loss_function(X, p, M, config):
    loss_ = {}
    
    """build weighted adjacency matrix of DAG"""
    B = build_dag(Y, p, M, config)
    
    if config["equal_variances"]:
        nll = 0.5 * config["d"] * torch.log(torch.pow(X - X @ B, 2).sum())
        nll -= torch.log(torch.abs(torch.linalg.det(torch.eye(config["d"]) - B)))
    else: # non equal variances
        raise ValueError("Equal variances assumption is required.")
    loss_["nll"] = nll
    
    # sparsity loss
    L1 = torch.linalg.norm(B, ord=1)
    loss_['L1'] = L1

    loss = nll + config["lambda"] * L1
    loss_['loss'] = loss
    
    return loss, loss_, B
#%%
def main():
    config = vars(get_args(debug=True)) # default configuration
    config["cuda"] = torch.cuda.is_available()
    wandb.config.update(config)
    
    set_random_seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["cuda"]:
        torch.cuda.manual_seed(config["seed"])
    
    """load dataset"""   
    dataset = SyntheticDataset(config)
    X = dataset.X
    # center the dataset
    X = X - X.mean(axis=0, keepdims=True)
    X = torch.FloatTensor(X)
    if config["cuda"]:
        X = X.cuda()
    B_true = dataset.B
    B_bin_true = dataset.B_bin
    
    wandb.run.summary['W_true'] = wandb.Table(data=pd.DataFrame(B_true))
    fig = viz_graph(B_true.round(2), size=(7, 7), show=config["fig_show"])
    wandb.log({'Graph': wandb.Image(fig)})
    fig = viz_heatmap(B_true.round(2), size=(5, 4), show=config["fig_show"])
    wandb.log({'heatmap': wandb.Image(fig)})
    
    p = torch.randn((config["d"], ), 
                    requires_grad=True)
    M = torch.randn((config["d"], config["d"]), 
                    requires_grad=True) # set diagonals to be zero
    optimizer = torch.optim.Adam([p, M], lr=config["lr"])
    
    for iteration in range(config["epochs"]):
        optimizer.zero_grad()
        loss, loss_, B = loss_function(X, p, M, config)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            h = h_function(B, config)
            loss_['h(B)'] = h
        
        if iteration % 500 == 0:
            print_input = "[iteration {:03d}]".format(iteration)
            print_input += ''.join([', {}: {:.4f}'.format(x, y.item()) for x, y in loss_.items()])
            print(print_input)
        
        """update log"""
        wandb.log({x : y.item() for x, y in loss_.items()})
    
    """post-process"""
    B_hat = B.detach().numpy().copy()
    B_hat[np.abs(B_hat) < config["w_threshold"]] = 0.
    B_hat = B_hat.astype(float).round(2)
    
    fig = viz_graph(B_hat, size=(7, 7), show=config["fig_show"])
    wandb.log({'Graph_est': wandb.Image(fig)})
    fig = viz_heatmap(B_hat, size=(5, 4), show=config["fig_show"])
    wandb.log({'heatmap_est': wandb.Image(fig)})

    wandb.run.summary['Is DAG?'] = is_dag(B_hat)
    wandb.run.summary['B_hat'] = wandb.Table(data=pd.DataFrame(B_hat))
    B_diff = (B_true.astype(float).round(2) - B_hat).astype(float).round(2)
    B_diff = (B_diff != 0).astype(float).round(2)
    fig = viz_graph(B_diff, size=(7, 7), show=config["fig_show"])
    wandb.log({'Graph_diff': wandb.Image(fig)})
    wandb.run.summary['W_diff'] = wandb.Table(data=pd.DataFrame(B_diff))

    """accuracy"""
    B_true_ = (B_true != 0).astype(float)
    B_hat_ = (B_hat != 0).astype(float)
    acc = count_accuracy(B_true_, B_hat_)
    wandb.log(acc)
    
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%