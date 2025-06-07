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
        self.ReLU = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_dim, input_dim)

        nn.ModuleList([self.layer_1, self.layer_2, self.ReLU])
    
    def forward(self, x):
        out = self.layer_1(x)
        out = self.ReLU(out)
        out = self.layer_2(out)
        return out

class EncoderBlock(nn.Module):
    def __init__(self, input_dim:int, num_attention_heads:int, attention_out_dim:int, feedforward_hiiden_dim:int):
        super().__init__()
        self.multiheadattention = MultiHeadAttentionBlock(input_dim, attention_out_dim, num_attention_heads)
        self.add_norm = AddandNorm(input_dim)
        self.feedforward = FeedForwardBlock(input_dim, feedforward_hiiden_dim)

        nn.ModuleList([self.multiheadattention, self.add_norm, self.feedforward])
    
    def forward(self, x):
        multi_out = self.multiheadattention(x)
        add_norm_out_1 = self.add_norm(x, multi_out)
        feed_forward_out = self.feedforward(add_norm_out_1)
        add_norm_out_2 = self.add_norm(add_norm_out_1, feed_forward_out)
        return add_norm_out_2

class Encoder(nn.Module):
    def __init__(self, input_dim:int, num_blocks:int, num_attention_heads:int, attention_out_dim:int, feedforward_hidden_dim:int):
        super().__init__()
        self.encoder_block_heads = nn.ModuleList([EncoderBlock(input_dim, num_attention_heads, attention_out_dim, feedforward_hidden_dim) for _ in range(num_blocks)])
        
        nn.ModuleList([self.encoder_block_heads])
    
    def forward(self, x):
        for block in self.encoder_block_heads:
            x = block(x)
        return x

class MaskedSelfAttention(nn.Module):
    def __init__(self, input_dim:int, output_dim:int):
        super().__init__()
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(-1)

        nn.ModuleList([self.query, self.key, self.value, self.softmax])
    
    def forward(self, x, mask):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        key_T = key.transpose(1, 2)

        intermidiate_1 = torch.matmul(query, key_T)/math.sqrt(key.shape[-1])
        intermidiate_2 = intermidiate_1 + mask
        intermidiate_3 = self.softmax(intermidiate_2)
        out = torch.matmul(intermidiate_3, value)

        return out

class MultiHeadMaskedAttention(nn.Module):
    def __init__(self, input_dim:int, output_dim:int, num_heads:int):
        super().__init__()
        self.heads = nn.ModuleList([MaskedSelfAttention(input_dim, output_dim) for _ in range(num_heads)])
        self.W0 = nn.Linear(output_dim*num_heads, input_dim)

        nn.ModuleList([self.heads])
    
    def forward(self, x):
        outputs = [head(x) for head in self.heads]
        out = torch.cat(outputs, -1)
        out = self.W0(out)
        return out

class CrossAttention(nn.Module):
    def __init__(self, encoder_output_dim:int, output_dim:int, masked_attention_dim:int):
        super().__init__()
        self.value = nn.Linear(encoder_output_dim, output_dim)
        self.key = nn.Linear(encoder_output_dim, output_dim)
        self.query = nn.Linear(masked_attention_dim, output_dim)
        self.softmax = nn.Softmax(-1)

        nn.ModuleList([self.value, self.key, self.query, self.softmax])
    
    def forward(self, encoder_output:torch.Tensor, x:torch.Tensor):
        query = self.query(x)
        key = self.key(encoder_output)
        value = self.value(encoder_output)

        key_T = key.transpose(1, 2)
        intermidiate_1 = torch.matmul(query, key_T)/math.sqrt(key.shape[-1])
        intermidiate_2 = self.softmax(intermidiate_1)
        out = torch.matmul(intermidiate_2, value)
        return out

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, encoder_output_dim:int, output_dim:int, masked_attention_dim:int, num_heads:int):
        super().__init__()
        self.blocks = nn.ModuleList([CrossAttention(encoder_output_dim, output_dim, masked_attention_dim) for _ in range(num_heads)])
        self.W0 = nn.Linear(output_dim*num_heads, masked_attention_dim)

        nn.ModuleList([self.blocks, self.W0])
    
    def forward(self, encoder_output:torch.Tensor, x:torch.Tensor):
        outputs = [block(encoder_output, x) for block in self.blocks]
        out = torch.cat(outputs, -1)
        out = self.W0(out)
        return out

class DecoderBlock(nn.Module):
    def __init__(self, masked_input_dim:int, masked_output_dim:int, num_masked_heads:int, encoder_output_dim:int, cross_attention_output_dim:int, num_cross_attention_heads:int, feed_forward_dim:int):
        super().__init__()
        self.MultiHeadMaskedAttention = MultiHeadMaskedAttention(masked_input_dim, masked_output_dim, num_masked_heads)
        self.AddandNorm = AddandNorm(masked_input_dim)
        self.MultiHeadCrossAttention = MultiHeadCrossAttention(encoder_output_dim, cross_attention_output_dim, masked_input_dim, num_cross_attention_heads)
        self.feedforward = FeedForwardBlock(masked_input_dim, feed_forward_dim)

        nn.ModuleList([self.MultiHeadMaskedAttention, self.AddandNorm, self.MultiHeadCrossAttention, self.feedforward])
    
    def forward(self, x, encoder_output):
        masked_output = self.MultiHeadMaskedAttention(x)
        add_norm_1 = self.AddandNorm(x, masked_output)

        cross_output = self.MultiHeadCrossAttention(encoder_output, add_norm_1)
        add_norm_2 = self.AddandNorm(add_norm_1, cross_output)

        feedforward = self.feedforward(add_norm_2)
        add_norm_3 = self.AddandNorm(add_norm_2, feedforward)

        return add_norm_3

class Decoder(nn.Module):
    def __init__(self, masked_input_dim:int, masked_output_dim:int, num_masked_heads:int, encoder_output_dim:int, cross_attention_output_dim:int, num_cross_attention_heads:int, feed_forward_dim:int, num_blocks:int):
        super().__init__()
        self.decoder_block_heads = nn.ModuleList([DecoderBlock(masked_input_dim, masked_output_dim, num_masked_heads, encoder_output_dim, cross_attention_output_dim, num_cross_attention_heads, feed_forward_dim) for _ in range(num_blocks)])

        nn.ModuleList([self.decoder_block_heads])
    
    def forward(self, x, encoder_output):
        for block in self.decoder_block_heads:
            x = block(x, encoder_output)
        return x
