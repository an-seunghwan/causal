#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.autograd import Variable
#%%
class Encoder(nn.Module):
    def __init__(self, 
                 config,
                 adj_A,
                 hidden_dim,
                 tol=0.1):
        super(Encoder, self).__init__()
        
        self.config = config
        self.adj_A = nn.Parameter(Variable(torch.from_numpy(adj_A).float(), requires_grad=True))
        self.fc1 = nn.Linear(config["x_dim"], hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, config["x_dim"], bias=True)
        self.Wa = nn.Parameter(torch.zeros(config["x_dim"]), requires_grad=True)
        self.z = nn.Parameter(torch.tensor(tol))
        self.z_positive = nn.Parameter(torch.ones_like(torch.from_numpy(adj_A)).float())
        self.init_weights()
    
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def normalize_adjacency_matrix(self, adj):
        return torch.inverse(torch.eye(adj.shape[0]).float() - adj.transpose(0, 1))
        
    def forward(self, input):
        if torch.sum(self.adj_A != self.adj_A):
            print('nan error\n')
        
        # to amplify values of A and accelerate convergence
        adj_A_amplified = torch.sinh(3. * self.adj_A)
        adj_A_normalized = self.normalize_adjacency_matrix(adj_A_amplified)
        
        h = F.relu(self.fc1(input))
        h = self.fc2(h)
        logits = torch.matmul(adj_A_normalized, h + self.Wa) - self.Wa
        return h, logits, adj_A_normalized
#%%
# hidden_dim = 32
# adj_A = nn.Parameter(Variable(torch.from_numpy(adj_A).float(), requires_grad=True))
# fc1 = nn.Linear(config["x_dim"], hidden_dim, bias=True)
# fc2 = nn.Linear(hidden_dim, config["x_dim"], bias=True)
# Wa = nn.Parameter(torch.zeros(config["x_dim"]), requires_grad=True)

# # to amplify values of A and accelerate convergence
# adj_A_amplified = torch.sinh(3. * adj_A)

# def normalize_adjacency_matrix(adj):
#     return torch.inverse(torch.eye(adj.shape[0]).float() - adj.transpose(0, 1))

# adj_A_normalized = normalize_adjacency_matrix(adj_A_amplified)
# h = F.relu(fc1(train_batch))
# h = fc2(h)
# logits = torch.matmul(adj_A_normalized, h + Wa) - Wa
#%%