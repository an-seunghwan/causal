#%%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import numpy as np
import pandas as pd
import tqdm

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
    project="(causal-proposal)inequality", 
    entity="anseunghwan",
    tags=["linear"],
)
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--seed', type=int, default=18, 
                        help='seed for repeatable results')
    parser.add_argument('--n', default=1000, type=int,
                        help='the number of dataset')
    parser.add_argument('--d', default=10, type=int,
                        help='the number of nodes')
    parser.add_argument('--s0', default=15, type=int,
                        help='expected number of edges')
    parser.add_argument('--graph_type', type=str, default='ER',
                        help='graph type: ER, SF, BP')
    parser.add_argument('--sem_type', type=str, default='gauss',
                        help='sem type: gauss, exp, gumbel, uniform, logistic, poisson')

    parser.add_argument('--t', default=0.01, type=float,
                        help='t')
    parser.add_argument('--mu', default=5, type=float,
                        help='mu')
    parser.add_argument('--delta', default=1, type=float,
                        help='delta')
    
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate')
    parser.add_argument('--max_iter', default=100, type=int,
                        help='maximum iteration')
    parser.add_argument('--max_epoch', default=10000, type=int,
                        help='maximum iteration')
    parser.add_argument('--tol', default=1e-2, type=float,
                        help='tolerance')
    parser.add_argument('--w_tol', default=1e-2, type=float,
                        help='w value tolerance')
    parser.add_argument('--w_threshold', default=0.3, type=float,
                        help='weight adjacency matrix threshold')
    parser.add_argument('--lambda', default=0.1, type=float,
                        help='weight of LASSO regularization')
    
    parser.add_argument('--fig_show', default=True, type=bool)

    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def h_fun(W):
    """Evaluate DAGness constraint"""
    h = trace_expm(W * W) - W.shape[0]
    return h

def loss_function(X, W_est, t, config):
    """centering step"""
    loss_ = []
    
    recon = 0.5 / config["n"] * torch.sum(torch.pow(X - X.matmul(W_est), 2))
    loss_.append(('recon', recon))
    
    L1 = config["lambda"] * torch.norm(W_est, p=1)
    loss_.append(('L1', L1))
    
    h = h_fun(W_est)
    log_barrier = - torch.log(- h + config["delta"])
    loss_.append(('log_barrier', log_barrier))
        
    loss = t * (recon + L1) + log_barrier
    loss_.append(('loss', loss))
    
    return loss, loss_
#%%
# def main():
config = vars(get_args(debug=True)) # default configuration
wandb.config.update(config)

'''simulate DAG and weighted adjacency matrix'''
set_random_seed(config["seed"])
B_true = simulate_dag(config["d"], config["s0"], config["graph_type"])
W_true = simulate_parameter(B_true)

wandb.run.summary['W_true'] = wandb.Table(data=pd.DataFrame(W_true))
fig = viz_graph(W_true, size=(7, 7), show=config["fig_show"])
wandb.log({'Graph': wandb.Image(fig)})
fig = viz_heatmap(W_true, size=(5, 4), show=config["fig_show"])
wandb.log({'heatmap': wandb.Image(fig)})
#%%
'''simulate dataset'''
X = simulate_linear_sem(W_true, config["n"], config["sem_type"], normalize=True)
n, d = X.shape
assert n == config["n"]
assert d == config["d"]
wandb.run.summary['data'] = wandb.Table(data=pd.DataFrame(X))

'''optimization process'''
X = torch.FloatTensor(X)
#%%
"""barrier method"""
m = 1. # number of inequality constraints
t = config["t"] # initial value

# strictly feasible weighted adjacency matrix
W_est = torch.zeros((config["d"], config["d"]), 
                    requires_grad=True)
W_est_old = None

optimizer = torch.optim.Adam([W_est], lr=config["lr"])
#%%
for iteration in range(config["max_iter"]):
    logs = {
        'loss': [], 
        'recon': [],
        'L1': [],
        'log_barrier': [],
    }
    
    with torch.no_grad():
        W_est_old = W_est.detach().clone()
        
    # with tqdm.tqdm(total=config["max_epoch"]) as pbar:
    # while True:
    for _ in tqdm.tqdm(range(config["max_epoch"])):
        optimizer.zero_grad()
        loss, loss_ = loss_function(X, W_est, t, config)
        loss.backward()
        optimizer.step()
        
        # pbar.update(1)
        
        """inner stopping rule: no change in weight estimation (convergence)"""
        with torch.no_grad():
            h = h_fun(W_est).item()
            w_change = torch.sum(torch.abs(W_est_old - W_est)).item()
        if w_change < config["w_tol"]: 
            break
        else:
            W_est_old = W_est.detach().clone()

    """accumulate losses"""
    for x, y in loss_:
        logs[x] = logs.get(x) + [y.item()]
    
    print_input = "[iteration {:03d}]".format(iteration)
    print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y).round(2)) for x, y in logs.items()])
    print_input += ', h(W): {:.8f}'.format(h)
    print_input += ', t: {:.1f}'.format(t)
    print(print_input)
    
    """update log"""
    wandb.log({x : np.mean(y) for x, y in logs.items()})
    wandb.log({'h(W)' : h})
    wandb.log({'t' : t})
    
    """stopping criterion"""
    if m/t < config["tol"]:
        break
    else:
        t = config["mu"] * t
#%%
"""chech DAGness of estimated weighted graph"""
W_est = W_est.detach().numpy().astype(float)
W_est[np.abs(W_est) < config["w_threshold"]] = 0.
W_est = W_est.round(2)

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

fig = viz_graph(W_diff_, size=(7, 7), show=config["fig_show"])
wandb.log({'Graph_diff': wandb.Image(fig)})

"""accuracy"""
B_est = (W_est != 0).astype(float)
acc = count_accuracy(B_true, B_est)
wandb.log(acc)
#%%
wandb.run.finish()
#%%
# if __name__ == '__main__':
#     main()
# #%%