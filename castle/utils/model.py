#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from torch.autograd import Variable
#%%
class CASTLE(nn.Module):
    def __init__(self, config):
        super(CASTLE, self).__init__()
        
        self.config = config
        
        self.W1 = nn.ParameterList(
            [nn.Parameter(Variable(torch.randn(self.config["d"], self.config["hidden_dim"]) * 0.1, requires_grad=True))
            for _ in range(self.config["d"])])
        
        self.masks = [torch.transpose(1. - F.one_hot(torch.tensor([i] * self.config["hidden_dim"]), self.config["d"]), 0, 1) 
                        for i in range(self.config["d"])]

        self.fc2 = nn.Linear(self.config["hidden_dim"], self.config["hidden_dim"])

        self.W3 = nn.ParameterList(
            [nn.Parameter(Variable(torch.randn(self.config["hidden_dim"], 1) * 0.1, requires_grad=True))
            for _ in range(self.config["d"])])
        self.b3 = nn.ParameterList(
            [nn.Parameter(Variable(torch.randn(1, ) * 0.1, requires_grad=True))
            for _ in range(self.config["d"])])
    
    def build_adjacency_matrix(self):
        W1_masked = [w * m for w, m in zip(self.W1, self.masks)]
        W = torch.cat([torch.sqrt(torch.sum(torch.pow(w, 2), dim=1, keepdim=True)) for w in W1_masked], dim=1)
        return W
    
    def forward(self, input):
        W1_masked = [w * m for w, m in zip(self.W1, self.masks)]
        h = [torch.matmul(input, w) for w in W1_masked]
        h = [nn.ReLU()(self.fc2(h_)) for h_ in h]
        h = [nn.ReLU()(torch.matmul(h_, w) + b) for h_, w, b in zip(h, self.W3, self.b3)]
        h = torch.cat(h, dim=1)
        return h, W1_masked
#%%
def main():
    config = {
        "n": 1000,
        "d": 10,
        "hidden_dim": 32,
    }
    
    model = CASTLE(config)
    for x in model.parameters():
        print(x)
    
    batch = torch.randn(config["n"], config["d"])
    recon = model(batch)
    assert recon.shape == (config["n"], config["d"])
    
    B = model.build_adjacency_matrix()
    assert B.shape == (config["d"], config["d"])
    
    print("Model test pass!")
#%%
if __name__ == '__main__':
    main()
#%%
# W1 = [nn.Parameter(Variable(torch.randn(config["d"], config["hidden_dim"]) * 0.1, requires_grad=True))
#       for _ in range(config["d"])]
# masks = [torch.transpose(1. - F.one_hot(torch.tensor([i] * config["hidden_dim"]), config["d"]), 0, 1) 
#         for i in range(config["d"])]

# fc2 = nn.Linear(config["hidden_dim"], config["hidden_dim"])

# W3 = [nn.Parameter(Variable(torch.randn(config["hidden_dim"], 1) * 0.1, requires_grad=True))
#       for _ in range(config["d"])]
# b3 = [nn.Parameter(Variable(torch.randn(1, ) * 0.1, requires_grad=True))
#       for _ in range(config["d"])]

# W1_masked = [w * m for w, m in zip(W1, masks)]
# X = torch.randn(config["n"], config["d"])
# h = [torch.matmul(X, w) for w in W1_masked]
# h = [fc2(h_) for h_ in h]
# h = [nn.ReLU()(h_) for h_ in h]
# h = [nn.ReLU()(torch.matmul(h_, w) + b) for h_, w, b in zip(h, W3, b3)]
# h = torch.cat(h, dim=1)

# W = torch.cat([torch.sum(w, dim=1, keepdim=True) for w in W1_masked], dim=1)
#%%