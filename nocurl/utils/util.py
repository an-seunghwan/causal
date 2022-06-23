#%%
import numpy as np
import torch
import torch.nn.functional as F
#%%
def build_dag(p, M, config):
    Y = torch.zeros((config["d"], config["d"]))
    for i in range(config["d"]):
        for j in range(config["d"]):
            Y[i, j] = p[j] - p[i]
    ReLU_Y = torch.nn.ReLU()(Y)
    W = 0.5 * (M - M.t())
    B = W * ReLU_Y
    return B
#%%