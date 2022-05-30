#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
#%%
class LocallyConnected(nn.Module):
    """Local linear layer (applied to each node(variable))
    Args:
        num_linear: num of local linear layers (= num of nodes)
        in_features: m1
        out_features: m2
        bias: whether to include bias or not
    Shape:
        - Input: [n, d, m1]
        - Output: [n, d, m2]
    Attributes:
        weight: [d, m1, m2]
        bias: [d, m2]
    """
    
    def __init__(self, num_linear, input_features, output_features, bias=False):
        super(LocallyConnected, self).__init__()
        self.num_linear = num_linear
        self.input_features = input_features
        self.output_features = output_features
        
        self.weight = nn.Parameter(torch.Tensor(num_linear,
                                                input_features,
                                                output_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_linear,
                                                  output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    @torch.no_grad()
    def reset_parameters(self):
        k = 1. / self.input_features
        bound = math.sqrt(k)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input: torch.Tensor):
        # [n, d, 1, m2] = [n, d, 1, m1] @ [1, d, m1, m2]
        h = torch.matmul(input.unsqueeze(dim=2), self.weight.unsqueeze(dim=0))
        h = h.squeeze(dim=2)
        if self.bias is not None:
            # [n, d, m2] += [d, m2]
            h += self.bias
        return h
    
    def extra_repr(self):
        # (Optional) Set the extra information about this module. 
        # You can test it by printing an object of this class.
        return 'num_linear={}, in_features={}, out_features={}, bias={}'.format(
            self.num_linear, self.input_features, self.output_features,
            self.bias is not None
        )
#%%
class NotearsMLP(nn.Module):
    def __init__(self, d, hidden_dims, bias=False):
        super(NotearsMLP, self).__init__()
        assert hidden_dims[-1] == 1
        self.d = d
        self.hidden_dims = hidden_dims
        
        self.fc1 = nn.Linear(d, d * hidden_dims[0], bias=bias) # m1 = hidden_dims[0]
        layers = []
        for i in range(len(hidden_dims)-1):
            layers.append(LocallyConnected(d, hidden_dims[i], hidden_dims[i+1], bias=bias))
        self.fc2 = nn.ModuleList(layers)
    
    def forward(self, x):
        h = self.fc1(x)
        h = h.reshape(-1, self.d, self.hidden_dims[0])
        for layer in self.fc2:
            h = torch.sigmoid(h)
            h = layer(h)
        h = h.squeeze(dim=2)
        return h
    
    def h_func(self):
        """Constrain L2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        W = self.fc1.weight.view(self.d, -1, self.d) # [i, m1, k]
        W = torch.sum(W * W, axis=1) # [j, k]
        W = W.t() # [k, j]
        h = trace_expm(W) - self.d
        return h
    
    def l2_reg(self):
        """Take L2-norm-squared of all parameters"""
        reg = 0.
        reg += torch.sum(self.fc1.weight ** 2)
        for layer in self.fc2:
            reg += torch.sum(layer.weight ** 2)
        return reg

    def l1_reg(self):
        """Take L2-norm of fc1 weight"""
        return torch.sum(torch.abs(self.fc1.weight))
    
    @torch.no_grad()
    def get_adjacency(self):
        W = self.fc1.weight.view(self.d, -1, self.d) # [i, m1, k]
        W = torch.sum(W * W, axis=1) # [j, k]
        W = W.t() # [k, j]
        W = torch.sqrt(W)
        return W.cpu().detach().numpy()
#%%
d = config["d"]
hidden_dim = config["hidden_dim"]
num_layers = config["num_layers"]

batch.shape

"""without bias"""
bias = True
layers = []
in_dim = d
out_dim = hidden_dim
for i in range(num_layers):
    if i == num_layers - 1:
        out_dim = 1 # dimension of each node == 1
    layers.append(LocallyConnected(d, in_dim, out_dim, bias=bias))
    if i == 0:
        in_dim = hidden_dim
MLP = nn.ModuleList(layers)
#%%
"""masking layer"""
mask = torch.ones(d, d) - torch.eye(d)
mask = torch.stack([torch.diag(mask[i]) for i in range(d)], dim=0)

x = batch.unsqueeze(dim=1) 
h = x.repeat((1, d, 1)) # [n, d, d], m1 = d
h = torch.matmul(h.unsqueeze(dim=2), mask.unsqueeze(dim=0)) # [n, d, 1, d] = [n, d, 1, d] @ [1, d, d, d]
h = h.squeeze(dim=2)
h.shape

for i, layer in enumerate(MLP):
    h = layer(h)
    if i != config["num_layers"] - 1:
        h = F.leaky_relu(h)
h = h.squeeze(dim=2)
h.shape
#%%
prod = mask # [d, d, d]
for weight in MLP.parameters():
    if len(weight.shape) < 3: continue # bias
    w = torch.abs(weight.unsqueeze(dim=0))
    prod = torch.matmul(prod.unsqueeze(dim=2), w) # [d, d, 1, m2] = [d, d, 1, m1] @ [1, d, m1, m2]
    prod = prod.squeeze(dim=2)
prod = torch.sum(prod, axis=-1) # [j, i]
prod.t() # adjacency matrix
#%%