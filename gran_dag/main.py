#%%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import numpy as np
import pandas as pd
import tqdm

import torch
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
# import cdt
# from cdt.utils.R import RPackages, launch_R_script

from utils.simulation import (
    # is_dag,
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
    tags=["nonlinear"],
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
    parser.add_argument('--s0', default=15, type=int,
                        help='expected number of edges')
    parser.add_argument('--graph_type', type=str, default='ER',
                        help='graph type: ER, SF, BP')
    parser.add_argument('--sem_type', type=str, default='mim',
                        help='sem type: mlp, mim, gp, gp-add')
    parser.add_argument('--normalize_data', type=bool, default=True,
                        help='normalize dataset')

    parser.add_argument('--rho', default=1, type=float,
                        help='rho')
    parser.add_argument('--alpha', default=0, type=float,
                        help='alpha')
    parser.add_argument('--h', default=np.inf, type=float,
                        help='h')
    
    parser.add_argument("--num_layers", default=2, type=int,
                        help="hidden dimensions for MLP")
    parser.add_argument("--hidden_dim", default=10, type=int,
                        help="hidden dimensions for MLP")
    
    parser.add_argument('--batch_size', default=128, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate')
    parser.add_argument('--max_iter', default=1000, type=int,
                        help='maximum iteration')
    parser.add_argument('--train_iter', default=300, type=int,
                        help='maximum iteration')
    parser.add_argument('--h_tol', default=1e-8, type=float,
                        help='h value tolerance')
    parser.add_argument('--progress_rate', default=0.25, type=float,
                        help='progress rate')
    parser.add_argument('--rho_rate', default=10, type=float,
                        help='rho rate')
    
    parser.add_argument('--normalize_W', type=str, default='path',
                        help='normalize weighed adjacency matrix, "none" does not normalize')
    parser.add_argument('--square_W', type=bool, default=False,
                        help='if True, connectivity matrix is computed with absolute, otherwise, with square')
    parser.add_argument('--jacobian', type=bool, default=True,
                        help='use Jacobian in thresholding')
    
    parser.add_argument('--pns_threshold', type=float, default=0.75,
                        help='threshold in PNS')
    parser.add_argument('--num_neighbors', type=int, default=None,
                        help='number of neighbors to select in PNS')
    parser.add_argument('--edge_clamp_range', type=float, default=0.0001,
                        help='as we train, clamping the edges (i,j) to zero when prod_ij is that close to zero. '
                            '0 means no clamping. Uses masks on inputs. Once an edge is clamped, no way back.')
    
    parser.add_argument('--fig_show', default=False, type=bool)

    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def h_fun(W):
    """Evaluate DAGness constraint"""
    h = trace_expm(W * W) - W.shape[0]
    return h

def loss_function(batch, model, alpha, rho, config):
    """Evaluate value and gradient of augmented Lagrangian."""
    loss_ = []
    
    xhat = model(batch)
    W = model.get_adjacency(norm=config["normalize_W"], 
                            square=config["square_W"])
    
    nll = 0.5 * (model.logvar + torch.pow(xhat - batch, 2) / torch.max(torch.exp(model.logvar), torch.tensor(1e-8)))
    nll = torch.mean(torch.sum(nll, axis=1))
    loss_.append(('nll', nll))
    
    h = h_fun(W)
    aug = (0.5 * rho * (h ** 2))
    aug += (alpha * h)
    loss_.append(('aug', aug))
        
    loss = sum([y for _, y in loss_])
    loss_.append(('loss', loss))
    
    return loss, loss_
#%%
def is_dag(W):
    prod = np.eye(W.shape[0])
    for _ in range(W.shape[0]):
        prod = np.matmul(W, prod)
        if np.trace(prod) != 0: return False
    return True
#%%
def main():
    config = vars(get_args(debug=False)) # default configuration
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
    data_loader, X = simulate_nonlinear_sem(
        B_true, 
        config["n"], 
        config["sem_type"], 
        config["batch_size"],
        config["normalize_data"]
    )
    train_loader = iter(data_loader)
    
    if config["cuda"]:
        data_loader.cuda()

    model = GraNDAG(d=config["d"], 
                    hidden_dim=config["hidden_dim"],
                    num_layers=config["num_layers"])
    if config["cuda"]:
        model.cuda()
        
    """Preliminary Neighborhood Selection"""
    if config["d"] >= 50:
        print("Preliminary Neighborhood Selection...\n")
        x = X.detach().numpy().copy()
        model_mask = model.mask.detach().cpu().numpy()
        for node in tqdm.tqdm(range(config["d"]), desc="PNS"):
            x_other = x.copy()
            x_other[:, node] = 0.
            reg = ExtraTreesRegressor(n_estimators=500)
            reg = reg.fit(x_other, x[:, node])
            if config["num_neighbors"] is None:
                num_neighbors = config["d"]
            else:
                num_neighbors = config["num_neighbors"]
            selected_reg = SelectFromModel(
                reg, 
                threshold="{}*mean".format(config["pns_threshold"]),
                prefit=True,
                max_features=num_neighbors
            )
            mask_selected = selected_reg.get_support(indices=False).astype(float)
            model_mask[:, node] *= mask_selected
        with torch.no_grad():
            model.mask.copy_(torch.Tensor(model_mask))

    """optimization process"""
    print("optimization process...\n")
    model.train()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config["lr"])
    
    # initial values
    rho = config["rho"]
    alpha = config["alpha"]
    h = config["h"]
    
    for iteration in range(config["max_iter"]):
        logs = {
                'loss': [], 
                'nll': [],
                'aug': [],
            }
        for epoch in tqdm.tqdm(range(config["train_iter"]), desc="primal update"):
            
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
                w_adj = model.get_adjacency(norm=config["normalize_W"], square=config["square_W"])
                to_keep = (w_adj > config["edge_clamp_range"]).type(torch.Tensor)
                model.mask *= to_keep
                
                h_new = h_fun(w_adj).item()
        else:
            with torch.no_grad():
                w_adj = model.get_adjacency(norm=config["normalize_W"], square=config["square_W"])
                h_new = h_fun(w_adj).item()
        
        print_input = "[iteration {:03d}]".format(iteration)
        print_input += ''.join([', {}: {:.4f}'.format(x, round(np.mean(y), 2)) for x, y in logs.items()])
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
        to_keep = (w_adj > 0).type(torch.Tensor)
        model.mask *= to_keep
    
    """to DAG"""
    print("to DAG...\n")
    model.eval()
    if config["jacobian"]:
        jac_avg = torch.zeros(config["d"], config["d"])
        X.requires_grad = True
        xhat = model(X)
        nll = 0.5 * (model.logvar + torch.pow(xhat - X, 2) / torch.max(torch.exp(model.logvar), torch.tensor(1e-8)))
        nll = torch.unbind(nll, axis=1)
        for node in range(config["d"]):
            jac = torch.autograd.grad(nll[node], X, retain_graph=True, grad_outputs=torch.ones(X.shape[0]))[0]
            jac_avg[node, :] = torch.abs(jac).mean(0)
        W = jac_avg.t()
    else:
        W = model.get_adjacency(norm=config["normalize_W"], square=config["square_W"])
    
    W = W.detach().cpu().numpy()
    with torch.no_grad():
        thresholds = np.unique(W)
        for step, t in enumerate(thresholds):
            to_keep = torch.Tensor(W > t + 1e-8)
            new_mask = model.mask * to_keep
            if is_dag(new_mask):
                model.mask.copy_(new_mask)
                break
    
    # """CAM pruning"""
    
    """retrain"""
    print("retrain...\n")
    model.train()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config["lr"])
    
    for iteration in range(config["train_iter"]):
        try:
            [batch] = next(train_loader)
        except:
            train_loader = iter(data_loader)
            [batch] = next(train_loader)

        """optimization step on augmented lagrangian"""
        optimizer.zero_grad()
        xhat = model(batch)
        nll = 0.5 * (model.logvar + torch.pow(xhat - batch, 2) / torch.max(torch.exp(model.logvar), torch.tensor(1e-8)))
        nll = torch.mean(torch.sum(nll, axis=1))
        loss = nll
        loss.backward()
        optimizer.step()
        
        print_input = "[iteration {:03d}]".format(iteration)
        print_input += ', {}: {:.4f}'.format('nll', round(nll.item(), 2))
        print(print_input)
        
        """update log"""
        wandb.log({'nll' : nll.item()})
        
    """chech DAGness of estimated weighted graph"""
    # W_est = model.get_adjacency(norm=config["normalize_W"], square=config["square_W"])
    # W_est = W_est.detach().numpy().astype(float).round(3)
    W_est = model.mask
    W_est = W_est.detach().numpy().astype(float)

    fig = viz_graph(W_est, size=(7, 7), show=config['fig_show'])
    wandb.log({'Graph_est': wandb.Image(fig)})
    fig = viz_heatmap(W_est, size=(5, 4), show=config['fig_show'])
    wandb.log({'heatmap_est': wandb.Image(fig)})

    wandb.run.summary['Is DAG?'] = is_dag(W_est)
    wandb.run.summary['W_est'] = wandb.Table(data=pd.DataFrame(W_est))

    B_est = W_est
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