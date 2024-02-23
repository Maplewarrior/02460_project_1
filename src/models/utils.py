import torch.nn as nn

class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)
    
class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape  # Target shape

    def forward(self, x):
        return x.view((-1, ) + self.shape)  # (-1, ) infers the batch size
