import torch
from torch import nn

training = True

if training == True:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

class Postional_Encoding(nn.Module):
    def __init__(self, encoding_size:int):
        super().__init__()
        self.d_model = encoding_size
        nn.ModuleList([self.d_model])
    
    def forward(self, pos_max:int, x):
        pos_tensor = torch.arange(0, pos_max, 1, dtype=torch.float32, device=device)
        i_values = torch.arange(0, (self.d_model/2), 1, device=device)
        sin_values = torch.sin(pos_tensor.view(-1, 1) / (10000**(2*i_values/self.d_model)))
        cos_values = torch.cos(pos_tensor.view(-1, 1) / (10000**(((2*i_values) + 1)/self.d_model)))
        
        stacked = torch.stack((sin_values, cos_values), dim=2)
        interleaved = stacked.view(sin_values.size(0), -1)
        out = x + interleaved
        return out

class InputBlock(nn.Module):
    def __init__(self, vocabulary_size:int, embedding_size:int):
        self.embedding_layer = nn.Embedding(vocabulary_size, embedding_size)
        self.positional_encoding = Postional_Encoding(embedding_size)
        nn.ModuleList([self.embedding_layer, self.positional_encoding])
    
    def forward(self, x):
        pos_max = x.shape[-1]
        out = self.embedding_layer(x)
        out = self.positional_encoding(pos_max, out)
        return out