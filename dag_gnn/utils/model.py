#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from torch.autograd import Variable
#%%
class Encoder(nn.Module):
    def __init__(self, 
                 config,
                 adj_A,
                 hidden_dim):
        super(Encoder, self).__init__()
        
        self.config = config
        self.adj_A = nn.Parameter(Variable(torch.from_numpy(adj_A).float(), requires_grad=True))
        self.fc1 = nn.Linear(config["x_dim"], hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, config["x_dim"], bias=True)
        self.Wa = nn.Parameter(torch.zeros(config["x_dim"]), requires_grad=True)
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def normalize_adjacency_matrix(self, adj):
        return torch.eye(adj.shape[0]).float() - adj.transpose(0, 1)
    
    def amplified_adjacency_matrix(self):
        return torch.sinh(3. * self.adj_A)
        
    def forward(self, input):
        if torch.sum(self.adj_A != self.adj_A):
            print('nan error\n')
        
        # to amplify values of A and accelerate convergence
        adj_A_amplified = torch.sinh(3. * self.adj_A)
        # (I - A^T)
        adj_A_normalized = self.normalize_adjacency_matrix(adj_A_amplified)
        
        h = F.relu(self.fc1(input))
        h = self.fc2(h)
        logits = torch.matmul(adj_A_normalized, h + self.Wa) - self.Wa
        return logits, h, adj_A_amplified
#%%
class Decoder(nn.Module):
    def __init__(self, 
                 config,
                 hidden_dim):
        super(Decoder, self).__init__()
        
        self.config = config
        self.fc1 = nn.Linear(config["x_dim"], hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, config["x_dim"], bias=True)
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def inverse_normalize_adjacency_matrix(self, adj):
        return torch.inverse(torch.eye(adj.shape[0]).float() - adj.transpose(0, 1))
        
    def forward(self, input, adj_A_amplified, Wa):
        adj_A_inverse_normalized = self.inverse_normalize_adjacency_matrix(adj_A_amplified)
        h = torch.matmul(adj_A_inverse_normalized, input + Wa) - Wa
        recon = F.relu(self.fc1(h))
        recon = self.fc2(recon)
        return recon, h
#%%
def main():
    config = {
        "d": 5,
        "x_dim": 3,
    }
    
    b = 128
    adj_A = np.zeros((config["d"], config["d"]))
    
    encoder = Encoder(config, adj_A, 32)
    # for x in encoder.parameters():
    #     print(x)
    h, logits, adj_A_amplified = encoder(torch.ones(b, config["d"], config["x_dim"]))
    assert h.shape == (b, config["d"], config["x_dim"])
    assert logits.shape == (b, config["d"], config["x_dim"])
    assert adj_A_amplified.shape == (config["d"], config["d"])
    print("Encoder test pass!")
    
    decoder = Decoder(config, 32)
    # for x in decoder.parameters():
    #     print(x)
    recon, z = decoder(logits, adj_A_amplified, encoder.Wa)
    assert recon.shape == (b, config["d"], config["x_dim"])
    assert z.shape == (b, config["d"], config["x_dim"])
    print("Decoder test pass!")
#%%
if __name__ == '__main__':
    main()
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
# fc1 = nn.Linear(config["x_dim"], hidden_dim, bias=True)
# fc2 = nn.Linear(hidden_dim, config["x_dim"], bias=True)

# def inverse_normalize_adjacency_matrix(adj):
#     return torch.inverse(torch.eye(adj.shape[0]).float() - adj.transpose(0, 1))

# adj_A_inverse_normalized = inverse_normalize_adjacency_matrix(adj_A_amplified)
# z = torch.matmul(adj_A_inverse_normalized, logits + Wa) - Wa
# h = F.relu(fc1(z))
# h = fc2(h)
#%%