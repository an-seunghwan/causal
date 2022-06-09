#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#%%
class MCSL(nn.Module):
    def __init__(self, config):
        super(MCSL, self).__init__()
        self.config = config
        
        # if config["model_type"] == "nn":
        nets = {}
        for i in range(config["d"]):
            layers = []
            for j in range(config["num_layer"]):
                input_dim = config["hidden_dim"]
                if j == 0:
                    input_dim = config["d"]
                layers.append(nn.Linear(input_dim, config["hidden_dim"]))
                layers.append(nn.LeakyReLU(negative_slope=0.05))
            layers.append(nn.Linear(config["hidden_dim"], 1))
            nets[str(i)] = nn.Sequential(*layers)
        self.nets = nn.ModuleDict(nets)        
        
        self.tau = config["temperature"]

        w = torch.nn.init.uniform_(torch.Tensor(config["d"], config["d"]),
                                a=-1e-10, b=1e-10)
        self.w = torch.nn.Parameter(w)

    """remove all seed options"""
    def sample_gumbel(self, shape, eps=1e-20):
        u = torch.rand(shape)
        u = -torch.log(-torch.log(u + eps) + eps)
        u[np.arange(shape[0]), np.arange(shape[0])] = 0 # set diagonal to zero
        return u

    def gumbel_sigmoid(self, logits, tau):
        gumbel_softmax_sample = logits + self.sample_gumbel(logits.shape) - self.sample_gumbel(logits.shape)
        return torch.sigmoid(gumbel_softmax_sample / self.tau)

    def _preprocess_graph(self, w, tau):
        w_prob = self.gumbel_sigmoid(w, tau)
        w_prob = w_prob * (1. - torch.eye(w.shape[0])) # set diagonal to zeros
        return w_prob

    def forward(self, input):
        w_prime = self._preprocess_graph(self.w, self.tau)
        xhat = []
        for i in range(self.config["d"]):
            mask = w_prime[:, i].unsqueeze(dim=0)
            xhat.append(self.nets[str(i)](input * mask))
        return torch.cat(xhat, dim=1), w_prime
#%%
def main():
    config = {
        "n": 100,
        "d": 7,
        "num_layer": 3,
        "hidden_dim": 16,
        "temperature": 0.2
    }
    
    model = MCSL(config)
    print(model)
    
    batch = torch.rand(config["n"], config["d"])
    recon = model(batch)
    assert recon.shape == (config["n"], config["d"])
    
    print("Model test pass!")
#%%
if __name__ == '__main__':
    main()
#%%
# nets = {}
# for i in range(config["d"]):
#     layers = []
#     for j in range(config["num_layer"]):
#         input_dim = config["hidden_dim"]
#         if j == 0:
#             input_dim = config["d"]
#         layers.append(nn.Linear(input_dim, config["hidden_dim"]))
#         layers.append(nn.LeakyReLU(negative_slope=0.05))
#     layers.append(nn.Linear(config["hidden_dim"], 1))
#     nets[str(i)] = nn.Sequential(*layers)
# nets = nn.ModuleDict(nets)

# tau = config["temperature"]

# w = torch.nn.init.uniform_(torch.Tensor(config["d"], config["d"]),
#                            a=-1e-10, b=1e-10)
# w = torch.nn.Parameter(w)

# """remove all seed options"""
# def sample_gumbel(shape, eps=1e-20):
#     u = torch.rand(shape)
#     u = -torch.log(-torch.log(u + eps) + eps)
#     u[np.arange(shape[0]), np.arange(shape[0])] = 0 # set diagonal to zero
#     return u

# def gumbel_sigmoid(logits, tau):
#     gumbel_softmax_sample = logits + sample_gumbel(logits.shape) - sample_gumbel(logits.shape)
#     return torch.sigmoid(gumbel_softmax_sample / tau)

# def _preprocess_graph(w, tau):
#     w_prob = gumbel_sigmoid(w, tau)
#     w_prob = w_prob * (1. - torch.eye(w.shape[0])) # set diagonal to zeros
#     return w_prob

# w_prime = _preprocess_graph(w, tau)

# xhat = []
# for i in range(config["d"]):
#     mask = w[:, i].unsqueeze(dim=0)
#     xhat.append(nets[str(i)](X * mask))
# torch.cat(xhat, dim=1)
#%%