#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.autograd import Variable
#%%
adj_A = nn.Parameter()
#%%
# if torch.sum(self.adj_A != self.adj_A):
#     print('nan error\n')
#%%