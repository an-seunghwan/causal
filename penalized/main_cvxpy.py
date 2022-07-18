#%%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import numpy as np
import pandas as pd
import tqdm
import networkx as nx

import scipy.stats
from pyglmnet import GLMCV
import asgl

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
    project="(causal)penalized", 
    entity="anseunghwan",
    tags=["linear", "equal-var"],
)
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results')
    parser.add_argument('--n', default=100, type=int,
                        help='the number of dataset')
    parser.add_argument('--d', default=50, type=int,
                        help='the number of nodes')
    parser.add_argument('--degree', default=6, type=int,
                        help='degree of graph')
    parser.add_argument('--graph_type', type=str, default='ER',
                        help='graph type: ER or SF')
    parser.add_argument('--noise_type', type=str, default='gaussian_ev',
                        help='noise type: gaussian_ev, gaussian_nv, exponential, gumbel')
    parser.add_argument('--B_scale', type=float, default=1,
                        help='scaling factor for range of B')

    # parser.add_argument('--lr', default=1e-2, type=float,
    #                     help='learning rate')
    # parser.add_argument('--epochs', default=10000, type=int,
    #                     help='number of iterations for training')
    # parser.add_argument('--tol', default=1e-8, type=float,
    #                     help='stopping criterion tolerance')
    parser.add_argument('--alpha1', default=0.9, type=float,
                        help='choice of tuining parameter for initial weight')
    parser.add_argument('--alpha2', default=0.1, type=float,
                        help='choice of tuining parameter for adaptive LASSO weight')
    
    parser.add_argument('--w_threshold', default=0.0001, type=float,
                        help='Threshold used to determine whether has edge in graph, element greater'
                            'than the w_threshold means has a directed edge, otherwise has not.')
    
    parser.add_argument('--fig_show', default=False, type=bool)

    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    config = vars(get_args(debug=False)) # default configuration
    wandb.config.update(config)
    set_random_seed(config["seed"])
    
    """load dataset"""   
    dataset = SyntheticDataset(config)
    X = dataset.X
    # center the dataset
    X = X - X.mean(axis=0, keepdims=True)
    X = X / X.std(axis=0, keepdims=True)
    B_true = dataset.B
    # B_bin_true = dataset.B_bin
    
    # G = nx.DiGraph(B_true)
    # list(nx.topological_sort(G))
    
    """check lower triangular matrix"""
    assert np.allclose(B_true, np.triu(B_true))
    
    wandb.run.summary['W_true'] = wandb.Table(data=pd.DataFrame(B_true))
    fig = viz_heatmap(np.flipud(B_true.round(2)), size=(5, 4), show=config["fig_show"])
    wandb.log({'heatmap': wandb.Image(fig)})
    
    """regular LASSO"""
    B_est = np.zeros((config["d"], config["d"]))
    for j in tqdm.tqdm(range(1, config["d"]), desc="regular LASSO"):
        L1_lambda = 2 * pow(config["n"], -1/2) * scipy.stats.norm.ppf(1 - config["alpha1"] / (2 * config["d"] * j))
        glm = GLMCV(distr="gaussian", 
                    alpha=1, # LASSO
                    reg_lambda=L1_lambda, 
                    solver="cdfast", 
                    fit_intercept=False)
        # glm = GLMCV(distr="gaussian", 
        #             alpha=1, # LASSO
        #             reg_lambda=np.logspace(-3, -2, base=10, num=10), 
        #             cv=10, # num of cv = 10
        #             solver="cdfast", 
        #             fit_intercept=False)
        glm.fit(X[:, :j], X[:, j])
        # glm.reg_lambda_opt_
        B_est[:j, j] = glm.beta_
    
    """adaptive weight"""
    weights = np.triu(
        np.maximum(1, np.nan_to_num(1 / np.abs(B_est), posinf=0)),
        k=1).astype(float)
    
    """adaptive LASSO"""
    B_est_adaptive = np.zeros((config["d"], config["d"]))
    for j in tqdm.tqdm(range(1, config["d"]), desc="adaptive LASSO"):
        L1_lambda = 2 * pow(config["n"], -1/2) * scipy.stats.norm.ppf(1 - config["alpha2"] / (2 * config["d"] * j))
        # cvxpy
        alasso = asgl.ASGL(model="lm", 
                            penalization="alasso", 
                            lambda1=L1_lambda, 
                            alpha=1, # without group LASSO
                            lasso_weights=weights[:j, j],
                            intercept=False)
        alasso.fit(X[:, :j], X[:, j])
        B_est_adaptive[:j, j] = alasso.coef_[0]
    
    """post-process"""
    B_hat = B_est_adaptive.copy()
    B_hat[np.abs(B_hat) < config["w_threshold"]] = 0.
    # B_hat = B_hat.astype(float).round(2)
    
    wandb.run.summary['B_hat'] = wandb.Table(data=pd.DataFrame(B_hat))    
    fig = viz_heatmap(np.flipud(B_hat), size=(5, 4), show=config["fig_show"])
    wandb.log({'heatmap_est': wandb.Image(fig)})

    # wandb.run.summary['Is DAG?'] = is_dag(B_hat)
    # B_diff = (B_true.astype(float).round(2) - B_hat).astype(float).round(2)
    # B_diff = (B_diff != 0).astype(float).round(2)
    # fig = viz_graph(B_diff, size=(7, 7), show=config["fig_show"])
    # wandb.log({'Graph_diff': wandb.Image(fig)})
    # wandb.run.summary['W_diff'] = wandb.Table(data=pd.DataFrame(B_diff))

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
    # B_est = np.zeros((config["d"], config["d"]))
    # for j in range(1, config["d"]):
    #     theta = torch.nn.init.normal_(torch.zeros((j, 1), requires_grad=True), 
    #                                 mean=0.0, std=0.1)
    #     optimizer = torch.optim.Adam([theta], lr=config["lr"])
        
    #     loss_old = np.inf
    #     for iteration in tqdm.tqdm(range(config["epochs"]), desc="regular LASSO of {}".format(j+1)):
    #         optimizer.zero_grad()
    #         loss, loss_ = loss_function(X, theta, j, config)
    #         loss.backward()
    #         optimizer.step()
            
    #         # if iteration % 100 == 0:
    #         #     print_input = "[iteration {:03d}]".format(iteration)
    #         #     print_input += ''.join([', {}: {:.4f}'.format(x, y.item()) for x, y in loss_.items()])
    #         #     print(print_input)
            
    #         """update log"""
    #         wandb.log({x : y.item() for x, y in loss_.items()})
            
    #         """stopping rule"""
    #         with torch.no_grad():
    #             if np.abs(loss_old - loss.detach().item()) < config["tol"]:
    #                 B_est[j, :j] = theta.detach().clone().t()
    #                 break
    #             loss_old = loss.detach().clone()
#%%
    # X = torch.FloatTensor(X)
    # B_est = np.zeros((config["d"], config["d"]))
    # for j in range(1, config["d"]):
    #     theta = torch.nn.init.normal_(torch.zeros((j, 1), requires_grad=True), 
    #                                 mean=0.0, std=0.1)
    #     optimizer = torch.optim.Adam([theta], lr=config["lr"])
        
    #     w = weights[j, :j]
        
    #     loss_old = np.inf
    #     for iteration in tqdm.tqdm(range(config["epochs"]), desc="adaptive LASSO of {}".format(j + 1)):
    #         optimizer.zero_grad()
    #         loss, loss_ = loss_function(X, theta, j, config, w)
    #         loss.backward()
    #         optimizer.step()
            
    #         # if iteration % 100 == 0:
    #         #     print_input = "[iteration {:03d}]".format(iteration)
    #         #     print_input += ''.join([', {}: {:.4f}'.format(x, y.item()) for x, y in loss_.items()])
    #         #     print(print_input)
            
    #         """update log"""
    #         wandb.log({x : y.item() for x, y in loss_.items()})
            
    #         """stopping rule"""
    #         with torch.no_grad():
    #             if np.abs(loss_old - loss.detach().item()) < config["tol"]:
    #                 B_est[j, :j] = theta.detach().t()
    #                 break
    #             loss_old = loss.detach()
#%%