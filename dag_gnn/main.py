#%%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import numpy as np
import pandas as pd

import torch
#%%
config = {
    'data_type': 'synthetic', # discrete, real
    "n": 5000,
    "d": 5,
    "degree": 2,
    "graph_type": "ER",
    "sem_type": "gauss",
    
    "lr": 0.001,
    "batch_size": 128,
    
    "rho": 1, # initial value
    "alpha": 0., # initial value
    "h": np.inf, # initial value
    
    "loss_type": 'l2',
    "max_iter": 100, 
    "h_tol": 1e-8, 
    "w_threshold": 0.3,
    "lambda": 0.005,
    "progress_rate": 0.25,
    "rho_max": 1e+16, 
    "rho_rate": 2.,
}
#%%