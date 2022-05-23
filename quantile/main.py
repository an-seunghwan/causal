#%%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    project="(causal)quantile", 
    entity="anseunghwan",
    tags=["proposal"],
    # name='notears'
)
#%%
import argparse
def get_args(debug=False):
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--seed', type=int, default=10, 
                        help='seed for repeatable results')
    parser.add_argument('--n', default=500, type=int,
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
    parser.add_argument('--max_iter', default=100, type=int,
                        help='maximum iteration')
    parser.add_argument('--h_tol', default=1e-8, type=float,
                        help='h value tolerance')
    parser.add_argument('--w_threshold', default=0.3, type=float,
                        help='weight adjacency matrix threshold')
    parser.add_argument('--lambda', default=0.001, type=float,
                        help='weight of LASSO regularization')
    parser.add_argument('--progress_rate', default=0.25, type=float,
                        help='progress rate')
    parser.add_argument('--rho_max', default=1e+16, type=float,
                        help='rho max')
    parser.add_argument('--rho_rate', default=10, type=float,
                        help='rho rate')
    
    parser.add_argument('--show_fig', default=False, type=bool)

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

def loss_function(X, W_est, alpha, rho, config, quantiles):
    """Evaluate value and gradient of augmented Lagrangian."""
    loss = 0
    for i in range(len(quantiles)):
        tau = quantiles[i]
        R = X - X.matmul(W_est[i])
        check = torch.max(tau * R, - (1. - tau) * R)
        loss += 1. / config["n"] * check.sum() + config["lambda"] * torch.norm(W_est[i], p=1)
        
        h = h_fun(W_est[i])
        loss += (0.5 * rho[i] * (h ** 2))
        loss += (alpha[i] * h)
    return loss
#%%
def main():
    config = vars(get_args(debug=False)) # default configuration
    wandb.config.update(config)

    """simulate DAG and weighted adjacency matrix"""
    set_random_seed(config["seed"])
    B_true = simulate_dag(config["d"], config["s0"], config["graph_type"])
    W_true = simulate_parameter(B_true)

    wandb.run.summary['W_true'] = wandb.Table(data=pd.DataFrame(W_true))
    fig = viz_graph(W_true, size=(7, 7), show=config["show_fig"])
    wandb.log({'Graph': wandb.Image(fig)})
    fig = viz_heatmap(W_true, size=(5, 4), show=config["show_fig"])
    wandb.log({'heatmap': wandb.Image(fig)})

    """simulate dataset"""
    X = simulate_linear_sem(W_true, config["n"], config["sem_type"], normalize=True)
    n, d = X.shape
    assert n == config["n"]
    assert d == config["d"]
    wandb.run.summary['data'] = wandb.Table(data=pd.DataFrame(X))

    """optimization process"""
    X = torch.FloatTensor(X)
    
    """quantiles"""
    quantiles = np.linspace(0.1, 0.9, 9)

    W_est = [torch.zeros((config["d"], config["d"]), requires_grad=True) 
             for _ in range(len(quantiles))]

    # initial values
    rho = [config["rho"] for _ in range(len(quantiles))]
    alpha = [config["alpha"] for _ in range(len(quantiles))]
    h = [config["h"] for _ in range(len(quantiles))]

    optimizer = torch.optim.Adam(W_est, lr=config["lr"])

    for iteration in range(config["max_iter"]):
        # primal update
        count = 0
        h_old = [np.inf for _ in range(len(quantiles))]
        while True:
            optimizer.zero_grad()
            loss = loss_function(X, W_est, alpha, rho, config, quantiles)
            loss.backward()
            optimizer.step()
            
            h_new = [h_fun(W).item() for W in W_est]
            # no change in weight estimation (convergence)
            if sum(abs(np.array(h_old) - np.array(h_new))) < 1e-8 * len(quantiles): 
                break
            h_old = h_new
                
            count += 1
            """update log"""
            wandb.log(
                {
                    # "inner_step": count,
                    "inner_loop/h": sum(h_new),
                    "inner_loop/loss": loss.item()
                }
            )
        
        # dual ascent step
        for i in range(len(quantiles)):
            alpha[i] += rho[i] * h_new[i]
        
        # stopping rules
        if sum(h_new) <= len(quantiles) * config["h_tol"]:
            # update
            h = h_new
            """update log"""
            wandb.log(
                {
                    # "iteration": iteration,
                    "train/h": sum(h),
                    "train/rho": sum(rho),
                    "train/alpha": sum(alpha),
                    "train/loss": loss.item()
                }
            )
            break
        else:
            """update log"""
            wandb.log(
                {
                    # "iteration": iteration,
                    "train/h": sum(h_new),
                    "train/rho": sum(rho),
                    "train/alpha": sum(alpha),
                    "train/loss": loss.item()
                }
            )
            for i in range(len(quantiles)):
                if h_new[i] > config["progress_rate"] * h[i]:
                    rho[i] *= config["rho_rate"]
                    if rho[i] >= config["rho_max"]:
                        break
            # update
            h = h_new
        
        print('[iteration {:03d}]: loss: {:.4f}, h(W): {:.4f}, primal update: {:04d}'.format(
            iteration, loss.item(), sum(h), count))

    """chech DAGness of estimated weighted graph"""
    W_est = [W.detach().numpy().astype(float).round(2) for W in W_est]
    for i in range(len(quantiles)):
        W_est[i][np.abs(W_est[i]) < config["w_threshold"]] = 0.

    def figure_to_array(fig):
        fig.canvas.draw()
        return np.array(fig.canvas.renderer._renderer)

    figs = [viz_graph(W, size=(7, 7)) for W in W_est]
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    for i in range(len(quantiles)):
        axes.flatten()[i].imshow(figure_to_array(figs[i]))
        axes.flatten()[i].set_title("quantile: {:.2f}".format(quantiles[i]))
        axes.flatten()[i].axis('off')
    plt.show()
    plt.close()
    wandb.log({'Graph_est': wandb.Image(fig)})
    
    figs = [viz_heatmap(W, size=(5, 4)) for W in W_est]
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    for i in range(len(quantiles)):
        axes.flatten()[i].imshow(figure_to_array(figs[i]))
        axes.flatten()[i].set_title("quantile: {:.2f}".format(quantiles[i]))
        axes.flatten()[i].axis('off')
    plt.show()
    plt.close()
    wandb.log({'heatmap_est': wandb.Image(fig)})

    wandb.run.summary['Is DAG?'] = [is_dag(W) for W in W_est]
    # wandb.run.summary['W_est'] = wandb.Table(data=pd.DataFrame(W_est))
    # wandb.run.summary['W_diff'] = wandb.Table(data=pd.DataFrame(W_true - W_est))

    # W_ = (W_true != 0).astype(float)
    # W_est_ = (W_est != 0).astype(float)
    # W_diff_ = np.abs(W_ - W_est_)

    # figs = [viz_graph(W, size=(7, 7)) for W in W_diff_]
    # fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    # for i in range(len(quantiles)):
    #     axes.flatten()[i].imshow(figure_to_array(figs[i]))
    #     axes.flatten()[i].set_title("quantile: {:.2f}".format(quantiles[i]))
    #     axes.flatten()[i].axis('off')
    # plt.show()
    # plt.close()
    # wandb.log({'Graph_diff': wandb.Image(fig)})

    """accuracy"""
    # B_est = (W_est != 0).astype(float)
    # acc = count_accuracy(B_true, B_est)
    # wandb.run.summary['acc'] = acc
    
    wandb.run.summary['config'] = config
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%