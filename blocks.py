import torch
from torch import nn
import math

training = True

if training == True:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

class Postional_Encoding(nn.Module):
    def __init__(self, encoding_size:int):
        super().__init__()
        self.d_model = encoding_size
    
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
        super().__init__()
        self.embedding_layer = nn.Embedding(vocabulary_size, embedding_size)
        self.positional_encoding = Postional_Encoding(embedding_size)
        nn.ModuleList([self.embedding_layer, self.positional_encoding])
    
    def forward(self, x):
        pos_max = x.shape[-1]
        out = self.embedding_layer(x)
        out = self.positional_encoding(pos_max, out)
        return out

class SelfAttentionHead(nn.Module):
    def __init__(self, input_size:int, output_size:int):
        super().__init__()
        self.query = nn.Linear(input_size, output_size)
        self.key = nn.Linear(input_size, output_size)
        self.value = nn.Linear(input_size, output_size)
        self.softmax = nn.Softmax(-1)

        nn.ModuleList([self.query, self.key, self.value, self.softmax])
    
    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        key_T = key.transpose(1, 2)
        intermidiate = torch.matmul(query, key_T)/math.sqrt(key.shape[-1])
        weights = self.softmax(intermidiate)

        contextual_embedding = torch.matmul(weights, value)

        return contextual_embedding

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, input_size:int, output_size:int, num_heads:int):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(input_size, output_size) for _ in range(num_heads)])
        self.W0 = nn.Linear(output_size*num_heads, input_size)

        nn.ModuleList([self.heads, self.W0])
    
    def forward(self, x):
        outputs = [layer(x) for layer in self.heads]
        out = torch.cat(outputs, -1)
        final_embeddings = self.W0(out)
        return final_embeddings

class AddandNorm(nn.Module):
    def __init__(self, input_dim:int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=input_dim)

        nn.ModuleList([self.layer_norm])
    
    def forward(self, previous_x, new_x):
        out = previous_x + new_x
        out = torch.reshape(out, (out.shape[0]*out.shape[1], out.shape[-1]))
        out = self.layer_norm(out)
        out = torch.reshape(out, previous_x.shape)
        return out

class FeedForwardBlock(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int):
        super().__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, input_dim)

        nn.ModuleList([self.layer_1, self.layer_2])
    
    def forward(self, x):
        out = self.layer_1(x)
        out = self.layer_2(out)
        return out