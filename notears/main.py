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
    project="(causal)NOTEARS", 
    entity="anseunghwan",
    tags=["linear"],
)
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results')
    parser.add_argument('--n', default=1000, type=int,
                        help='the number of dataset')
    parser.add_argument('--d', default=5, type=int,
                        help='the number of nodes')
    parser.add_argument('--s0', default=8, type=int,
                        help='expected number of edges')
    parser.add_argument('--graph_type', type=str, default='ER',
                        help='graph type: ER, SF, BP')
    parser.add_argument('--sem_type', type=str, default='gauss',
                        help='sem type: gauss, exp, gumbel, uniform, logistic, poisson')

    parser.add_argument('--rho', default=1, type=float,
                        help='rho')
    parser.add_argument('--alpha', default=0, type=float,
                        help='alpha')
    parser.add_argument('--h', default=np.inf, type=float,
                        help='h')
    
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate')
    parser.add_argument('--loss_type', type=str, default='l2',
                        help='loss type')
    parser.add_argument('--max_iter', default=100, type=int,
                        help='maximum iteration')
    parser.add_argument('--h_tol', default=1e-8, type=float,
                        help='h value tolerance')
    parser.add_argument('--w_threshold', default=0.3, type=float,
                        help='weight adjacency matrix threshold')
    parser.add_argument('--lambda', default=0.01, type=float,
                        help='weight of LASSO regularization')
    parser.add_argument('--progress_rate', default=0.25, type=float,
                        help='progress rate')
    parser.add_argument('--rho_max', default=1e+16, type=float,
                        help='rho max')
    parser.add_argument('--rho_rate', default=2, type=float,
                        help='rho rate')

    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()

# config = {
#     "seed": 10,
#     "n": 1000,
#     "d": 5,
#     "s0": 5,
#     "graph_type": 'ER',
#     "sem_type": 'gauss',
    
#     "rho": 1, # initial value
#     "alpha": 0., # initial value
#     "h": np.inf, # initial value
    
#     "lr": 0.001,
#     "loss_type": 'l2',
#     "max_iter": 100, 
#     "h_tol": 1e-8, 
#     "w_threshold": 0.2,
#     "lambda": 0.005,
#     "progress_rate": 0.25,
#     "rho_max": 1e+16, 
#     "rho_rate": 2.,
# }
#%%
def h_fun(W):
    """Evaluate DAGness constraint"""
    h = trace_expm(W * W) - W.shape[0]
    return h

def loss_function(X, W_est, alpha, rho, config):
    """Evaluate value and gradient of augmented Lagrangian."""
    loss_ = []
    
    R = X - X.matmul(W_est)
    recon = 0.5 / config["n"] * (R ** 2).sum()
    loss_.append(('recon', recon))
    
    L1 = config["lambda"] * torch.norm(W_est, p=1)
    loss_.append(('L1', L1))
    
    h = h_fun(W_est)
    aug = 0.5 * rho * (h ** 2)
    aug += alpha * h
    loss_.append(('aug', aug))
        
    loss = sum([y for _, y in loss_])
    loss_.append(('loss', loss))
    
    return loss, loss_
#%%
def main():
    config = vars(get_args(debug=True)) # default configuration
    wandb.config.update(config)

    '''simulate DAG and weighted adjacency matrix'''
    set_random_seed(config["seed"])
    B_true = simulate_dag(config["d"], config["s0"], config["graph_type"])
    W_true = simulate_parameter(B_true)

    wandb.run.summary['W_true'] = wandb.Table(data=pd.DataFrame(W_true))
    fig = viz_graph(W_true, size=(7, 7))
    # wandb.run.summary['Graph'] = wandb.Image(fig)
    wandb.log({'Graph': wandb.Image(fig)})
    fig = viz_heatmap(W_true, size=(5, 4))
    wandb.log({'heatmap': wandb.Image(fig)})
    # wandb.run.summary['heatmap'] = wandb.Image(fig)

    '''simulate dataset'''
    X = simulate_linear_sem(W_true, config["n"], config["sem_type"], normalize=True)
    n, d = X.shape
    assert n == config["n"]
    assert d == config["d"]
    wandb.run.summary['data'] = wandb.Table(data=pd.DataFrame(X))

    '''optimization process'''
    X = torch.FloatTensor(X)

    W_est = torch.zeros((config["d"], config["d"]), 
                        requires_grad=True)

    # initial values
    rho = config["rho"]
    alpha = config["alpha"]
    h = config["h"]

    optimizer = torch.optim.Adam([W_est], lr=config["lr"])

    for iteration in range(config["max_iter"]):
        """Perform one step of dual ascent in augmented Lagrangian."""
        
        logs = {
            'loss': [], 
            'recon': [],
            'L1': [],
            'aug': [],
        }
        
        # primal update
        h_old = np.inf
        while rho < config["rho_max"]:
            while True:
                optimizer.zero_grad()
                loss, loss_ = loss_function(X, W_est, alpha, rho, config)
                loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    h_new = h_fun(W_est).item()
                # no change in weight estimation (convergence)
                if abs(h_old - h_new) < 1e-8: 
                    break
                h_old = h_new
        
            """accumulate losses"""
            for x, y in loss_:
                logs[x] = logs.get(x) + [y.item()]
            
            with torch.no_grad():
                h_new = h_fun(W_est).item()
            if h_new > config["progress_rate"] * h:
                rho *= config["rho_rate"]
            else:
                break
        
        # dual ascent step
        h = h_new
        alpha += rho * h
        
        print_input = "[iteration {:03d}]".format(iteration)
        print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y).round(2)) for x, y in logs.items()])
        print_input += ', h(W): {:.8f}'.format(h)
        print(print_input)
        
        """update log"""
        wandb.log({x : np.mean(y) for x, y in logs.items()})
        wandb.log({'h(W)' : h})
        
        """stopping rule"""
        if h <= config["h_tol"] or rho >= config["rho_max"]:
            break
        
    """chech DAGness of estimated weighted graph"""
    W_est = W_est.detach().numpy().astype(float).round(2)
    W_est[np.abs(W_est) < config["w_threshold"]] = 0.

    fig = viz_graph(W_est, size=(7, 7))
    # wandb.run.summary['Graph_est'] = wandb.Image(fig)
    wandb.log({'Graph_est': wandb.Image(fig)})
    fig = viz_heatmap(W_est, size=(5, 4))
    # wandb.run.summary['heatmap_est'] = wandb.Image(fig)
    wandb.log({'heatmap_est': wandb.Image(fig)})

    wandb.run.summary['Is DAG?'] = is_dag(W_est)
    wandb.run.summary['W_est'] = wandb.Table(data=pd.DataFrame(W_est))
    wandb.run.summary['W_diff'] = wandb.Table(data=pd.DataFrame(W_true - W_est))

    W_ = (W_true != 0).astype(float)
    W_est_ = (W_est != 0).astype(float)
    W_diff_ = np.abs(W_ - W_est_)

    fig = viz_graph(W_diff_, size=(7, 7))
    # wandb.run.summary['Graph_diff'] = wandb.Image(fig)
    wandb.log({'Graph_diff': wandb.Image(fig)})

    """accuracy"""
    B_est = (W_est != 0).astype(float)
    acc = count_accuracy(B_true, B_est)
    wandb.log(acc)
    
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%