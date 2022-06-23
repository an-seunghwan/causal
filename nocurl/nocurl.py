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
    SyntheticDataset,
    count_accuracy
)

from utils.viz import (
    viz_graph,
    viz_heatmap,
)

from utils.util import (
    postprocess
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
    parser.add_argument('--degree', default=2, type=int,
                        help='degree of graph')
    parser.add_argument('--graph_type', type=str, default='ER',
                        help='graph type: ER or SF')
    parser.add_argument('--noise_type', type=str, default='gaussian_ev',
                        help='noise type: gaussian_ev, gaussian_nv, exponential, gumbel')
    parser.add_argument('--B_scale', type=float, default=1,
                        help='scaling factor for range of B')

    parser.add_argument('--w_threshold', default=0.3, type=float,
                        help='Threshold used to determine whether has edge in graph, element greater'
                            'than the w_threshold means has a directed edge, otherwise has not.')
    
    parser.add_argument('--fig_show', default=True, type=bool)

    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
def main():
    config = vars(get_args(debug=True)) # default configuration
    
    set_random_seed(config["seed"])
    torch.manual_seed(config["seed"])
    
    """load dataset"""   
    dataset = SyntheticDataset(config)
    X = dataset.X
    # center the dataset
    X = X - X.mean(axis=0, keepdims=True)
    X = torch.FloatTensor(X)
    B_true = dataset.B
    B_bin_true = dataset.B_bin
    
    fig = viz_graph(B_true.round(2), size=(7, 7), show=config["fig_show"])
    fig = viz_heatmap(B_true.round(2), size=(5, 4), show=config["fig_show"])
    
    
#%%
if __name__ == '__main__':
    main()
#%%