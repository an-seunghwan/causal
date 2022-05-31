#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
#%%
class GraNDAG(nn.Module):
    """GraNDAG model
    Args:
        d: num of nodes
        hidden_dim: hidden dimension of MLP
        num_layers: num of layers
        num_params: num of parameters of log likelihood
        bias: whether to include bias or not
    Shape:
        - Input: [n, d]
        - Output: [n, d]
    """
    
    def __init__(self, d, hidden_dim, num_layers, num_params=1, bias=False):
        super(GraNDAG, self).__init__()
        self.d = d
        self.hidden_dims = hidden_dim
        self.num_layers = num_layers
        self.num_params = num_params
        
        """without bias"""
        self.weights = []
        for i in range(num_layers):
            in_dim = out_dim = hidden_dim
            if i == 0:
                in_dim = d
            if i == num_layers - 1:
                out_dim = num_params
            self.weights.append(nn.Parameter(torch.Tensor(d,
                                                        out_dim,
                                                        in_dim)))

        """masking layer"""
        self.mask = torch.ones(d, d) - torch.eye(d)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        with torch.no_grad():
            for node in range(self.d):
                for i, w in enumerate(self.weights):
                    w = w[node]
                    nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain("leaky_relu"))
        
    def forward(self, x):
        h = x
        for k in range(self.num_layers):
            if k == 0:
                h = torch.einsum("tij, ljt, bj -> bti", self.weights[k], self.mask.unsqueeze(dim=0), h)
            else:
                h = torch.einsum("tij, btj -> bti", self.weights[k], h)
            if k != self.num_layers - 1:
                h = F.leaky_relu(h)
        h = h.squeeze(dim=2)
        return h
    
    def get_adjacency(self, norm="none", square=False):
        """Get weighted adjacency matrix"""
        # norm: normalize connectivity matrix
        prod = torch.eye(self.d)
        if norm != "none":
            prod_norm = torch.eye(self.d)
        for i, w in enumerate(self.weights):
            if square: 
                w = w ** 2
            else:
                w = torch.abs(w)
            if i == 0:
                prod = torch.einsum("tij, ljt, jk -> tik", w, self.mask.unsqueeze(dim=0), prod)
                if norm != "none":
                    tmp = 1. - torch.eye(self.d)
                    prod_norm = torch.einsum("tij, ljt, jk -> tik", torch.ones_like(w).detach(), tmp.unsqueeze(dim=0), prod_norm)
            else:
                prod = torch.einsum("tij, tjk -> tik", w, prod)
                if norm != "none":
                    prod_norm = torch.einsum("tij, tjk -> tik", torch.ones_like(w).detach(), prod_norm)

        # sum over density parameter axis
        prod = torch.sum(prod, axis=1)
        if norm == "path":
            prod_norm = torch.sum(prod_norm, axis=1)
            denominator = prod_norm + torch.eye(self.d) # avoid divide 0 on diagonals
            return (prod / denominator).t()
        elif norm == 'none':
            return prod.t()
        else:
            raise NotImplementedError
#%%
def main():
    n = 20
    d = 10
    hidden_dim = 8
    num_layers = 3
    
    model = GraNDAG(d, hidden_dim, num_layers)
    
    x = torch.rand(n, d)
    recon = model(x)
    assert recon.shape == (n, d)
    assert model.mask.shape == (d, d)
    assert model.get_adjacency().shape == (d, d)
    print('model test pass!')
#%%
if __name__ == "__main__":
    main()
#%%
# d = config["d"]
# hidden_dim = config["hidden_dim"]
# num_layers = config["num_layers"]
# num_params = 1

# [batch] = next(train_loader)
# batch.shape

# """without bias"""
# bias = False
# weights = []
# for i in range(num_layers):
#     in_dim = out_dim = hidden_dim
#     if i == 0:
#         in_dim = d
#     if i == num_layers - 1:
#         out_dim = num_params
#     weights.append(nn.Parameter(torch.Tensor(d,
#                                             out_dim,
#                                             in_dim)))

# print([w.shape for w in weights])
# """masking layer"""
# mask = torch.ones(d, d) - torch.eye(d)
# #%%
# h = batch
# for k in range(num_layers):
#     if k == 0:
#         h = torch.einsum("tij, ljt, bj -> bti", weights[k], mask.unsqueeze(dim=0), h)
#     else:
#         h = torch.einsum("tij, btj -> bti", weights[k], h)
#     if k != num_layers - 1:
#         h = F.leaky_relu(h)
# h = h.squeeze(dim=2)
# h.shape
# #%%
# norm = "none" # normalize connectivity matrix
# square = False

# prod = torch.eye(d)
# if norm != "none":
#     prod_norm = torch.eye(d)
# for i, w in enumerate(weights):
#     if square: 
#         w = w ** 2
#     else:
#         w = torch.abs(w)
#     if i == 0:
#         prod = torch.einsum("tij, ljt, jk -> tik", w, mask.unsqueeze(dim=0), prod)
#         if norm != "none":
#             tmp = 1. - torch.eye(d)
#             prod_norm = torch.einsum("tij, ljt, jk -> tik", torch.ones_like(w).detach(), tmp.unsqueeze(dim=0), prod_norm)
#     else:
#         prod = torch.einsum("tij, tjk -> tik", w, prod)
#         if norm != "none":
#             prod_norm = torch.einsum("tij, tjk -> tik", torch.ones_like(w).detach(), prod_norm)

# # sum over density parameter axis
# prod = torch.sum(prod, axis=1)
# if norm == "path":
#     prod_norm = torch.sum(prod_norm, axis=1)
#     denominator = prod_norm + torch.eye(d)
#     return (prod / denominator).t()
# elif norm == 'none':
#     return prod.t()
# else:
#     raise NotImplementedError
#%%