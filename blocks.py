import torch
from torch import nn

class Postional_Encoding(nn.Module):
    def __init__(self, encoding_size:int):
        super().__init__()
        self.d_model = encoding_size
    
    def forward(self, pos_max:int):
        pos_tensor = torch.arange(0, pos_max, 1, dtype=torch.float32)
        i_values = torch.arange(0, (self.d_model/2), 1)
        sin_values = torch.sin(pos_tensor.view(-1, 1) / (10000**(2*i_values/self.d_model)))
        cos_values = torch.cos(pos_tensor.view(-1, 1) / (10000**(((2*i_values) + 1)/self.d_model)))
        
        stacked = torch.stack((sin_values, cos_values), dim=2)
        interleaved = stacked.view(sin_values.size(0), -1)
        return interleaved