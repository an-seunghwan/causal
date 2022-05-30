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
    simulate_nonlinear_sem,
    count_accuracy,
)

from utils.model import (
    GraNDAG
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
    project="(causal)GRAN-DAG", 
    entity="anseunghwan",
    tags=["notears", "linear"],
    # name='notears'
)
#%%
import ast
def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results')
    parser.add_argument('--n', default=1000, type=int,
                        help='the number of dataset')
    parser.add_argument('--d', default=10, type=int,
                        help='the number of nodes')
    parser.add_argument('--s0', default=10, type=int,
                        help='expected number of edges')
    parser.add_argument('--graph_type', type=str, default='ER',
                        help='graph type: ER, SF, BP')
    parser.add_argument('--sem_type', type=str, default='gp-add',
                        help='sem type: mlp, mim, gp, gp-add')

    parser.add_argument('--rho', default=1e-3, type=float,
                        help='rho')
    parser.add_argument('--alpha', default=0, type=float,
                        help='alpha')
    parser.add_argument('--h', default=np.inf, type=float,
                        help='h')
    
    parser.add_argument("--num_layers", default=3, type=int,
                        help="hidden dimensions for MLP")
    parser.add_argument("--hidden_dim", default=8, type=int,
                        help="hidden dimensions for MLP")
    
    parser.add_argument('--batch_size', default=64, type=float,
                        help='learning rate')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate')
    parser.add_argument('--train_iter', default=1000, type=int,
                        help='maximum iteration')
    parser.add_argument('--h_tol', default=1e-8, type=float,
                        help='h value tolerance')
    parser.add_argument('--w_threshold', default=0.3, type=float,
                        help='weight adjacency matrix threshold')
    parser.add_argument('--progress_rate', default=0.9, type=float,
                        help='progress rate')
    parser.add_argument('--rho_rate', default=10, type=float,
                        help='rho rate')
    
    parser.add_argument('--edge-clamp-range', type=float, default=1e-4,
                        help='as we train, clamping the edges (i,j) to zero when prod_ij is that close to zero. '
                             '0 means no clamping. Uses masks on inputs. Once an edge is clamped, no way back.')
    
    parser.add_argument('--fig_show', default=True, type=bool)

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

def loss_function(batch, model, alpha, rho, config):
    """Evaluate value and gradient of augmented Lagrangian."""
    loss_ = []
    
    xhat = model(batch)
    W = model.get_adjacency()
    
    recon = 0.5 / config["n"] * torch.sum((xhat - batch) ** 2) 
    loss_.append(('recon', recon))
    
    h = h_fun(W)
    aug = (0.5 * rho * (h ** 2))
    aug += (alpha * h)
    loss_.append(('aug', aug))
        
    loss = sum([y for _, y in loss_])
    loss_.append(('loss', loss))
    
    return loss, loss_
#%%
def main():
    config = vars(get_args(debug=True)) # default configuration
    wandb.config.update(config)

    config["cuda"] = torch.cuda.is_available()
    set_random_seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["cuda"]:
        torch.cuda.manual_seed(config["seed"])
    
    '''simulate DAG and weighted adjacency matrix'''
    B_true = simulate_dag(config["d"], config["s0"], config["graph_type"])

    wandb.run.summary['W_true'] = wandb.Table(data=pd.DataFrame(B_true))
    fig = viz_graph(B_true, size=(7, 7), show=config["fig_show"])
    wandb.log({'Graph': wandb.Image(fig)})
    fig = viz_heatmap(B_true, size=(5, 4), show=config["fig_show"])
    wandb.log({'heatmap': wandb.Image(fig)})

    '''simulate dataset'''
    data_loader = simulate_nonlinear_sem(
        B_true, 
        config["n"], 
        config["sem_type"], 
        config["batch_size"]
    )
    train_loader = iter(data_loader)
    
    if config["cuda"]:
        data_loader.cuda()

    '''optimization process'''
    model = GraNDAG(d=config["d"], 
                    hidden_dim=config["hidden_dim"],
                    num_layers=config["num_layers"])
    if config["cuda"]:
        model.cuda()

    # initial values
    rho = config["rho"]
    alpha = config["alpha"]
    h = config["h"]

    optimizer = torch.optim.RMSprop(model.parameters(), lr=config["lr"])

    for iteration in range(config["train_iter"]):
        logs = {
            'loss': [], 
            'recon': [],
            'aug': [],
        }
        
        try:
            [batch] = next(train_loader)
        except:
            train_loader = iter(data_loader)
            [batch] = next(train_loader)

        """optimization step on augmented lagrangian"""
        optimizer.zero_grad()
        loss, loss_ = loss_function(batch, model, alpha, rho, config)
        loss.backward()
        optimizer.step()
        
        """accumulate losses"""
        for x, y in loss_:
            logs[x] = logs.get(x) + [y.item()]
        
        """clamp edges"""
        if config["edge_clamp_range"] != 0:
            with torch.no_grad():
                to_keep = (model.get_adjacency() > config["edge_clamp_range"]).type(torch.Tensor)
                model.mask *= torch.stack([torch.diag(to_keep[:, i]) for i in range(config["d"])], dim=0)
        
        with torch.no_grad():
            h_new = h_fun(model.get_adjacency()).item()
        
        print_input = "[iteration {:03d}]".format(iteration)
        print_input += ''.join([', {}: {:.4f}'.format(x, round(y[0], 2)) for x, y in logs.items()])
        print_input += ', h(W): {:.8f}'.format(h_new)
        print(print_input)
        
        """update log"""
        wandb.log({x : np.mean(y) for x, y in logs.items()})
        wandb.log({'h(W)' : h_new})
        
        """stopping rule"""
        if h_new > config["h_tol"]:
            """dual ascent step"""
            alpha += rho * h_new
            
            """Did the contraint improve sufficiently?"""
            if h_new > config["progress_rate"] * h:
                rho *= config["rho_rate"]
        else:
            break
        
        h = h_new
        
    """Final clamping of all edges"""
    with torch.no_grad():
        to_keep = (model.get_adjacency() > 0).type(torch.Tensor)
        model.mask *= torch.stack([torch.diag(to_keep[:, i]) for i in range(config["d"])], dim=0)
    
    """chech DAGness of estimated weighted graph"""
    W_est = model.get_adjacency().astype(float).round(2)
    W_est[np.abs(W_est) < config["w_threshold"]] = 0.

    fig = viz_graph(W_est, size=(7, 7), show=config['fig_show'])
    wandb.log({'Graph_est': wandb.Image(fig)})
    fig = viz_heatmap(W_est, size=(5, 4), show=config['fig_show'])
    wandb.log({'heatmap_est': wandb.Image(fig)})

    wandb.run.summary['Is DAG?'] = is_dag(W_est)
    wandb.run.summary['W_est'] = wandb.Table(data=pd.DataFrame(W_est))

    B_est = (W_est != 0).astype(float)
    W_diff_ = np.abs(B_true - B_est)

    fig = viz_graph(W_diff_, size=(7, 7), show=config['fig_show'])
    wandb.log({'Graph_diff': wandb.Image(fig)})

    """accuracy"""
    acc = count_accuracy(B_true, B_est)
    wandb.log(acc)
    
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%