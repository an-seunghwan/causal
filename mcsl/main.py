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
    DAG,
    IIDSimulation,
    count_accuracy
)

from utils.viz import (
    viz_graph,
    viz_heatmap,
)

from utils.model import (
    MCSL
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
    project="(causal)MCSL", 
    entity="anseunghwan",
    tags=["nonlinear"],
)
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
    parser.add_argument('--sem_type', type=str, default='gp',
                        help='sem type for linear method: gauss, exp, gumbel, uniform, logistic'
                            'sem type for nonlinear method: mlp, mim, gp, gp-add')
    parser.add_argument('--method', type=str, default='nonlinear',
                        help='causal structure type: linear, nonlinear')

    parser.add_argument('--model_type', type=str, default='nn',
                        help='nn denotes neural network, qr denotes quatratic regression.') # qr is not suppored yet
    parser.add_argument("--num_layer", default=4, type=int,
                        help="Number of hidden layer in neural network when model_type is nn")
    parser.add_argument("--hidden_dim", default=16, type=int,
                        help="Number of hidden dimension in hidden layer, when model_type is nn")
    
    parser.add_argument('--max_iter', default=25, type=int,
                        help='maximum number of iteration')
    parser.add_argument('--epochs', default=1000, type=int,
                        help='maximum iteration')
    parser.add_argument('--batch_size', default=100, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=0.03, type=float,
                        help='learning rate')
    parser.add_argument('--init_iter', default=2, type=int,
                        help='Initial iteration to disallow early stopping')
    parser.add_argument('--lambda', default=2e-3, type=float,
                        help='coefficient of LASSO penalty')
    
    parser.add_argument('--rho', default=1e-5, type=float,
                        help='rho')
    parser.add_argument('--rho_max', default=1e+14, type=float,
                        help='maximum rho value')
    parser.add_argument('--rho_rate', default=10, type=float,
                        help='Multiplication to amplify rho each time')
    parser.add_argument('--alpha', default=0, type=float,
                        help='alpha')
    parser.add_argument('--h', default=np.inf, type=float,
                        help='h')
    
    parser.add_argument('--h_tol', default=1e-10, type=float,
                        help='h value tolerance')
    parser.add_argument('--w_threshold', default=0.5, type=float,
                        help='Threshold used to determine whether has edge in graph, element greater'
                            'than the w_threshold means has a directed edge, otherwise has not.')
    parser.add_argument('--progress_rate', default=0.25, type=float,
                        help='progress rate')
    parser.add_argument('--temperature', default=0.2, type=float,
                        help='Temperature for gumbel sigmoid')
    
    parser.add_argument('--fig_show', default=True, type=bool)

    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def h_function(w, config):
    h_A = torch.trace(torch.matrix_exp(w * w)) - config["d"]
    return h_A

def loss_function(X, model, rho, alpha, config):
    model.train()
    
    loss_ = {}
    
    recon, w_prime = model(X)
    
    # reconstruction
    recon = 0.5 / config["n"] * torch.pow(recon - X, 2).sum() 
    loss_['recon'] = recon

    # sparsity loss
    L1 = config["lambda"] * torch.linalg.norm(w_prime, ord=1)
    loss_['L1'] = L1

    # augmentation and lagrangian loss
    """Evaluate DAGness constraint"""
    h_A = h_function(w_prime, config)
    aug = 0.5 * rho * (h_A ** 2)
    aug += alpha * h_A
    loss_['aug'] = aug
    
    loss = sum([y for _, y in loss_.items()])
    loss_['loss'] = loss
    
    return loss, loss_, h_A
#%%
def convert_logits_to_sigmoid(w, tau, config):
    sigmoid_w = torch.sigmoid(w / tau)
    sigmoid_w = sigmoid_w * (1. - torch.eye(config["d"]))
    return sigmoid_w
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
    dag_simulator = DAG()
    W_true = dag_simulator.erdos_renyi(
            n_nodes=config["d"],
            n_edges=config["s0"],
            weight_range=(0.5, 2.0)
    )
    W_true = (W_true != 0).astype(int)
    iid_simulator = IIDSimulation(
        W_true, 
        n=config["n"], 
        method=config["method"], 
        sem_type=config["sem_type"]
    )
    X = iid_simulator.X
    X = torch.FloatTensor(X)
    if config["cuda"]:
        X = X.cuda()
    
    wandb.run.summary['W_true'] = wandb.Table(data=pd.DataFrame(W_true))
    fig = viz_graph(W_true, size=(7, 7), show=config["fig_show"])
    wandb.log({'Graph': wandb.Image(fig)})
    fig = viz_heatmap(W_true, size=(5, 4), show=config["fig_show"])
    wandb.log({'heatmap': wandb.Image(fig)})
    
    model = MCSL(config)

    if config["cuda"]:
        model.cuda()

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config["lr"]
    )
    
    rho = config["rho"]
    alpha = config["alpha"]
    h = config["h"]
        
    for iteration in range(config["max_iter"]):
        logs = {
            'loss': [], 
            'recon': [],
            'L1': [],
            'aug': [],
        }
        
        """primal problem"""
        while rho < config["rho_max"]:
            # find argmin of primal problem (local solution) = update for config["epochs"] times
            for epoch in tqdm.tqdm(range(config["epochs"]), desc="primal update"):
                optimizer.zero_grad()
                loss, loss_, h_new = loss_function(X, model, rho, alpha, config)
                loss.backward()
                optimizer.step()
                
                """accumulate losses"""
                for x, y in loss_.items():
                    logs[x] = logs.get(x) + [y.item()]
                
            if h_new.item() > config["progress_rate"] * h:
                rho *= config["rho_rate"]
            else:
                break
        
        sigmoid_w = convert_logits_to_sigmoid(model.w.detach(), config["temperature"], config)
        h_logit = h_function(sigmoid_w, config)
        
        print_input = "[iteration {:03d}]".format(iteration)
        print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y).round(2)) for x, y in logs.items()])
        print_input += ', h(W): {:.8f}'.format(h_new.item())
        print_input += ', h(W_logit): {:.8f}'.format(h_logit.item())
        print(print_input)
        
        """update log"""
        wandb.log({x : np.mean(y) for x, y in logs.items()})
        wandb.log({'h(W)' : h_new.item()})
        wandb.log({'h(W_logit)' : h_logit.item()})
        
        """stopping rule"""
        if h_new.item() <= config["h_tol"] and iteration > config["init_iter"]:
            break
        
        """dual ascent"""
        h = h_new.detach().item()
        alpha += rho * h
    
    """final metrics"""
    w_est = convert_logits_to_sigmoid(model.w.detach(), config["temperature"], config)
    w_est[w_est <= config["w_threshold"]] = 0
    w_est[w_est > config["w_threshold"]] = 1
    w_est[np.arange(config["d"]), np.arange(config["d"])] = 0
    w_est = w_est.numpy()
    
    fig = viz_graph(w_est, size=(7, 7), show=config["fig_show"])
    wandb.log({'Graph_est': wandb.Image(fig)})
    fig = viz_heatmap(w_est, size=(5, 4), show=config["fig_show"])
    wandb.log({'heatmap_est': wandb.Image(fig)})

    wandb.run.summary['Is DAG?'] = is_dag(w_est)
    wandb.run.summary['w_est'] = wandb.Table(data=pd.DataFrame(w_est))
    fig = viz_graph(W_true - w_est, size=(7, 7), show=config["fig_show"])
    wandb.log({'Graph_diff': wandb.Image(fig)})
    wandb.run.summary['W_diff'] = wandb.Table(data=pd.DataFrame(W_true - w_est))

    """accuracy"""
    acc = count_accuracy(W_true, w_est)
    wandb.log(acc)
    
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%