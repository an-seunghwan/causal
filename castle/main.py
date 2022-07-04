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
    random_dag,
    swap_cols,
    swap_nodes,
    gen_data_nonlinear,
    count_accuracy
)

from utils.viz import (
    viz_graph,
    viz_heatmap,
)

from utils.model import (
    CASTLE
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
    project="(causal)CASTLE", 
    entity="anseunghwan",
    # tags=[],
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
    parser.add_argument('--branchf', default=4, type=int,
                        help='expected number of edges')
    parser.add_argument('--nonlinear_sigmoid', type=bool, default=True,
                        help='nonlinear causal structure type: nonlinear_1, nonlinear_2')

    parser.add_argument('--w_threshold', default=0.3, type=float,
                        help='threshold for weighted adjacency matrix')
    parser.add_argument('--lambda', default=1, type=float,
                        help='coefficient of supervised loss')
    parser.add_argument('--beta', default=1, type=float,
                        help='coefficient of LASSO penalty')
    parser.add_argument('--rho', default=1, type=float,
                        help='rho')
    parser.add_argument('--alpha', default=1, type=float,
                        help='alpha')
    
    parser.add_argument("--hidden_dim", default=32, type=int,
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
def h_fun(W):
    """Evaluate DAGness constraint"""
    h = trace_expm(W * W) - W.shape[0]
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
        
        recon, W1_masked = model(batch_x)
        W = model.build_adjacency_matrix()
        
        loss_ = []
        
        """prediction"""
        supervised_loss = torch.pow(batch_y - recon[:, 0], 2).sum() / config["batch_size"]
        loss_.append(('supervised_loss', supervised_loss))
        
        """reconstruction"""
        recon = torch.pow(recon - batch_x, 2).sum() 
        loss_.append(('recon', recon))

        """sparsity loss"""
        GroupL1 = sum([torch.norm(w, dim=1, p=2).sum() for w in W1_masked])
        loss_.append(('GroupL1', GroupL1))

        """augmentation and lagrangian loss"""
        h_A = h_fun(W)
        aug = 0.5 * config["rho"] * (h_A ** 2)
        aug += config["alpha"] * h_A
        loss_.append(('aug', aug))
        
        loss = (1. + config["lambda"] * config["rho"]) * supervised_loss
        loss += recon + config["beta"] * GroupL1 + aug
        loss_.append(('loss', loss))
        
        loss.backward()
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
    noise = random.uniform(0.3, 1.0)
    print("Setting noise to:", noise)
    
    G = random_dag(config["d"], config["d"] * config["branchf"])
    X = gen_data_nonlinear(G, var=noise, SIZE=config["n"], sigmoid=config["nonlinear_sigmoid"])
    
    for i in range(len(G.edges())):
        if len(list(G.predecessors(i))) > 0:
            X = swap_cols(X, str(0), str(i))
            G = swap_nodes(G, 0, i)
            break      
            
    #print("Number of parents of G", len(list(G.predecessors(i))))
    print("Edges = ", list(G.edges()))
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    B_true = np.array(nx.to_numpy_matrix(G))

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
        h = h_fun(W_est)
        
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