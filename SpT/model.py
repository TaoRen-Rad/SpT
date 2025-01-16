import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from .kan import nn.Linear

DTYPE = torch.float32

class TransformerWithLearnedPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        self.positional_embeddings = nn.Embedding(max_len, d_model)
    
    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        pos_embeddings = self.positional_embeddings(positions)
        return x + pos_embeddings

    
class SimpleMLP(nn.Module):
    def __init__(self, layer_sizes):
        super(SimpleMLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            # 除了最后一层外，每一层后面都加ReLU激活函数
            if i < len(layer_sizes) - 2:
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class SimpleConvolve(nn.Module):
    def __init__(self, channels, kernels, dropout=0.05):
        super(SimpleConvolve, self).__init__()
        self.convs = nn.ModuleList()
        
        self.first_channel = channels[0]
        for i in range(len(channels) - 1):
            padding = kernels[i] // 2
            self.convs.append(nn.Conv1d(channels[i], channels[i+1], kernel_size=kernels[i], padding=padding))

        self.dropout = nn.Dropout(dropout)
        # Calculate the number of features before the fully connected layer
        self.num_features = self._get_conv_output(1024)
        

    def forward(self, x):
        # batch, channel, seq_len
        x = x.reshape([x.shape[0], -1, x.shape[1]])
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x))
            x = F.max_pool1d(x, 2)
            if i != 0:  # Skip dropout on the first convolution layer
                x = self.dropout(x)
        return x
    
    def _get_conv_output(self, input_size):
        # Dummy pass to get the number of features
        input = torch.rand(1, self.first_channel, input_size)
        for conv in self.convs:
            input = F.relu(conv(input))
            input = F.max_pool1d(input, 2)
            # input = self.dropout(input)
        return input.size()

def estimate_unfold_length(input_length, patch_size, stride):
    """
    估计unfold操作后第二个维度的长度

    参数:
    input_length (int): 输入数据的长度
    patch_size (int): patch的大小
    stride (int): 步长

    返回:
    int: unfold操作后第二个维度的长度
    """
    if stride <= 0:
        raise ValueError("Stride must be a positive integer.")
    if patch_size > input_length:
        return 0
    
    unfold_length = (input_length - patch_size) // stride + 1
    return unfold_length

class MyModel(nn.Module):
    def __init__(self, patch_size, stride,
                 feature_dim, d_model, nhead, num_transformer_layers, 
                 dim_feedforward, final_mlp_layers):
        super(MyModel, self).__init__()
        
        # self.seq_len = self.simpleConvolve.num_features[2]
        seq_len = estimate_unfold_length(1024*3*2, patch_size, stride)
        print("Estimated seq_len: ", seq_len)
        self.positional_encoder = TransformerWithLearnedPositionalEmbedding(d_model=d_model, max_len=seq_len)
        
        # self.spectrum_linear = nn.Linear(self.window_size, self.window_size, dtype=dtype)
        
        self.patch_size = patch_size
        self.stride = stride
        
        self.radiance_linear = nn.Linear(patch_size, d_model)
        self.uniform_linear = nn.Linear(feature_dim, d_model)
        
        self.d_model = d_model
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))  # learnable [CLS] token
        
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, 
            nhead=nhead, dim_feedforward=dim_feedforward, dtype=DTYPE, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer, num_layers=num_transformer_layers, 
            enable_nested_tensor = False)
        
        # size = self.simpleConvolve.num_features
        print("MLP layers: ", [d_model] + final_mlp_layers)
        self.final_nn = SimpleMLP([d_model] + final_mlp_layers)
        # self.final_nn = SimpleMLP(final_mlp_layers)

    def forward(self, features: torch.tensor, radiances: torch.tensor):
        x = radiances.unfold(1, self.patch_size, self.stride)
        x = x.reshape([x.shape[0], -1, x.shape[-1]])
        x = self.radiance_linear(x)
        # x = self.uniform_linear(features).unsqueeze(1).repeat(1, x.size(1), 1) + x
        
        # Add [CLS] token at the beginning of the sequence
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) * self.uniform_linear(features).unsqueeze(1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = self.positional_encoder(x)
        x = x.permute(1, 0, 2)  # Transformer needs [seq_len, batch_size, d_model]
        
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # go back to [batch_size, seq_len, d_model]
        
        x = x[:, 0, :]  # take the [CLS] token's output
        x = self.final_nn(x)
        
        return x
