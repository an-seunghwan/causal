#%%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import numpy as np
import pandas as pd
import tqdm
import random
from sklearn.preprocessing import StandardScaler  

import torch
import networkx as nx

from utils.simulation import (
    set_random_seed,
    is_dag,
    simulate_dag,
    simulate_nonlinear_sem,
    count_accuracy
)

from utils.viz import (
    viz_graph,
    viz_heatmap,
)

from utils.model import (
    CASTLE
)

# from utils.trac_exp import trace_expm
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
    project="(causal)CASTLE", 
    entity="anseunghwan",
    tags=["nonlinear"],
)
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results')
    parser.add_argument('--data_type', type=str, default='synthetic',
                        help='types of data: synthetic, real')
    parser.add_argument('--n', default=1000, type=int,
                        help='the number of dataset')
    parser.add_argument('--d', default=10, type=int,
                        help='the number of nodes')
    # parser.add_argument('--branchf', default=4, type=int,
    #                     help='expected number of edges')
    # parser.add_argument('--nonlinear_sigmoid', type=bool, default=True,
    #                     help='nonlinear causal structure type: nonlinear_1, nonlinear_2')
    parser.add_argument('--degree', default=3, type=int,
                        help='expected number of edges')
    parser.add_argument('--graph_type', type=str, default='ER',
                        help='graph type: ER, SF')
    parser.add_argument('--sem_type', type=str, default='mim',
                        help='sem type: mlp, mim, gp, gp-add')

    parser.add_argument('--w_threshold', default=0.2, type=float,
                        help='threshold for weighted adjacency matrix')
    parser.add_argument('--lambda', default=1, type=float,
                        help='coefficient of supervised loss')
    parser.add_argument('--beta', default=2, type=float,
                        help='coefficient of LASSO penalty')
    parser.add_argument('--rho', default=10, type=float,
                        help='rho')
    parser.add_argument('--alpha', default=1, type=float,
                        help='alpha')
    
    parser.add_argument("--hidden_dim", default=8, type=int,
                        help="hidden dimensions for MLP")
    parser.add_argument("--num_layer", default=2, type=int,
                        help="hidden dimensions for MLP")
    
    parser.add_argument('--epochs', default=200, type=int,
                        help='maximum iteration')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate')
    
    parser.add_argument('--fig_show', default=False, type=bool)

    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def h_fun(W, config):
    # """Evaluate DAGness constraint"""
    # h = trace_expm(W * W) - W.shape[0]
    
    """truncated power series"""
    coff = 1.
    Z = W * W
    dag_I = config["d"]
    Z_in = torch.eye(config["d"])
    for i in range(1, 10):
        Z_in = torch.matmul(Z_in, Z)
        dag_I += 1./coff * torch.trace(Z_in)
        coff *= (i + 1)
    h = dag_I - config["d"]
    return h
#%%
def train(X, model, config, optimizer):
    model.train()
    
    logs = {
        'loss': [], 
        'supervised_loss': [],
        'recon': [],
        'GroupL1': [],
        'aug': [],
    }
    
    for i in range(X.shape[0] // config["batch_size"]):
        
        idxs = random.sample(range(X.shape[0]), config["batch_size"])
        batch_x = torch.FloatTensor(np.array(X[idxs, :]))
        batch_y = torch.FloatTensor(np.array(X[idxs, 0]))
        
        optimizer.zero_grad()
        
        xhat, W1_masked = model(batch_x)
        W = model.build_adjacency_matrix()
        # set diagnoal to zero
        # W *= 1. - torch.eye(config["d"])
        
        loss_ = []
        
        """prediction"""
        supervised_loss = torch.pow(batch_y - xhat[:, 0], 2).sum() / config["batch_size"]
        loss_.append(('supervised_loss', supervised_loss))
        
        """reconstruction"""
        recon = torch.pow(batch_x - xhat, 2).sum() 
        loss_.append(('recon', recon))

        """sparsity loss"""
        GroupL1 = sum([w.norm(dim=1, p=2).sum() for w in W1_masked])
        # GroupL1 = sum([torch.norm(w[:k, :], dim=1, p=2) for k, w in enumerate(W1_masked)])
        # GroupL1 += sum([torch.norm(w[k+1:, :], dim=1, p=2).sum() for k, w in enumerate(W1_masked)])
        loss_.append(('GroupL1', GroupL1))

        """augmentation and lagrangian loss"""
        h_A = h_fun(W, config)
        aug = 0.5 * config["rho"] * (h_A ** 2)
        aug += config["alpha"] * h_A
        loss_.append(('aug', aug))
        
        loss = (1. + config["lambda"] * config["rho"]) * supervised_loss
        loss += recon + config["beta"] * GroupL1 + aug
        loss_.append(('loss', loss))
        
        loss.backward()
        # nan gradient due to masking: set nan to zero
        for weight in model.parameters():
            weight.grad = torch.nan_to_num(weight.grad, nan=0.)
        optimizer.step()
        
        """accumulate losses"""
        for x, y in loss_:
            logs[x] = logs.get(x) + [y.item()]
    
    return logs
#%%
def main():
    config = vars(get_args(debug=False)) # default configuration
    config["cuda"] = torch.cuda.is_available()
    wandb.config.update(config)
    
    set_random_seed(config["seed"])
    torch.manual_seed(config["seed"])
    if config["cuda"]:
        torch.cuda.manual_seed(config["seed"])
        
    # noise = random.uniform(0.3, 1.0)
    # print("Setting noise to:", noise)
    
    # G = random_dag(config["d"], config["d"] * config["branchf"])
    # X = gen_data_nonlinear(G, SIZE=config["n"], sigmoid=config["nonlinear_sigmoid"])
    # X = gen_data_nonlinear(G, var=noise, SIZE=config["n"], sigmoid=config["nonlinear_sigmoid"])
    
    # print("Edges = ", list(G.edges()))
    
    '''simulate DAG and weighted adjacency matrix'''
    B_true = simulate_dag(config["d"], config["degree"], config["graph_type"])

    '''simulate dataset'''
    X, G, ordered_vertices = simulate_nonlinear_sem(B_true, config["n"], config["sem_type"])
    assert X.shape[0] == config["n"]
    assert X.shape[1] == config["d"]
    
    # standardization
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    min_parents = config["d"] // 2
    # target = ordered_vertices[-3]
    target = None
    while target is None:
        for i in range(config["d"]):
            if len(list(G.predecessors(i))) >= min_parents:
                target = i
                break  
        min_parents -= 1
    print("Number of parents of target:", min_parents)
    # target is the first column
    X = np.concatenate((X[:, [target]], X[:, :target], X[:, target + 1:]), axis=1)
    B_true = np.concatenate((B_true[:, [target]], B_true[:, :target], B_true[:, target + 1:]), axis=1)

    wandb.run.summary['W_true'] = wandb.Table(data=pd.DataFrame(B_true))
    fig = viz_graph(B_true, size=(7, 7), show=config["fig_show"])
    wandb.log({'Graph': wandb.Image(fig)})
    fig = viz_heatmap(B_true, size=(5, 4), show=config["fig_show"])
    wandb.log({'heatmap': wandb.Image(fig)})
    
    model = CASTLE(config)

    if config["cuda"]:
        model.cuda()

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config["lr"]
    )
    
    for epoch in range(config["epochs"]):
        logs = train(X, model, config, optimizer)
        
        W_est = model.build_adjacency_matrix().detach().data.clone()
        h = h_fun(W_est, config)
        
        if epoch % 20 == 0:
            print_input = "[iteration {:03d}]".format(epoch)
            print_input += ''.join([', {}: {:.4f}'.format(x, np.mean(y).round(2)) for x, y in logs.items()])
            print_input += ', h(W): {:.8f}'.format(h)
            print(print_input)
        
        """update log"""
        wandb.log({x : np.mean(y) for x, y in logs.items()})
        wandb.log({'h(W)' : h})
        
    """final metrics"""
    B = W_est.numpy()
    B[np.abs(B) < config["w_threshold"]] = 0.
    B = B.astype(float).round(2)
    B = (B != 0).astype(float)

    fig = viz_graph(B, size=(7, 7), show=config["fig_show"])
    wandb.log({'Graph_est': wandb.Image(fig)})
    fig = viz_heatmap(B, size=(5, 4), show=config["fig_show"])
    wandb.log({'heatmap_est': wandb.Image(fig)})

    wandb.run.summary['Is DAG?'] = is_dag(B)
    wandb.run.summary['B'] = wandb.Table(data=pd.DataFrame(B))
    wandb.run.summary['B_diff'] = wandb.Table(data=pd.DataFrame(B_true - B))

    """accuracy"""
    acc = count_accuracy(B_true, B)
    wandb.log(acc)
    
    wandb.run.finish()
#%%
if __name__ == '__main__':
    main()
#%%