import torch
from torch.nn import Module, Conv1d
from torch import Tensor
import torch.nn.functional as F
import math
import utils.ops as ops
import utils.kernel_functions as kops
from layers.kernel_transformer import KernelTransformer
from typing import Dict, Any
from .model_utils import summary


embed_config_basic = {
    'window_size': 10,
    'max_len': 2816,
}

default_kernerl_transformer_config = {
    'kernel_function': kops.ExponentialKernel(),  # This should be set separately as it's a Callable
    'd_model': 512,
    'nhead': 8,
    'num_layers': 16,
    'dim_feedforward': 2048,
    'dropout': 0.1,
    'activation': F.relu,
    'custom_encoder': None,
    'layer_norm_eps': 1e-5,
    'batch_first': True,
    'norm_first': True,
    'bias': True,
    'device': None,
    'dtype': None
}


def positional_encoding_sin_cos(embed_dim:int, max_seq_len:int) -> Tensor:
        position = torch.arange(max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        pos_encoding = torch.zeros(max_seq_len, embed_dim)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding

class KernelTransformerModel(Module):

    def __init__(self, transformer_config: Dict[str, Any], embed_config: Dict[str, Any]):
        super().__init__()
        self.transformer_config = transformer_config
        self.embedding_config = embed_config
        self.embeddings = self._create_embeddings()
        self.positional_encoding = self._create_positional_encodings()
        self.model = self._create_model()

    def _create_embeddings(self):
        return SignalEmbedding(embed_dim=self.transformer_config['d_model'],
                               device=self.transformer_config['device'],
                               dtype=self.transformer_config['dtype'],
                               **self.embedding_config)
                               
    def _create_positional_encodings(self):
        return positional_encoding_sin_cos(embed_dim=self.transformer_config['d_model'],
                                           max_seq_len=self.embeddings.expected_len
                                           )
    def _create_model(self):
        return KernelTransformer(**self.transformer_config)
    
    def forward(self, input):
        x = self.embeddings(input)
        assert x.size(-2) == self.positional_encoding.size(0),(
            "Mismatch in sequence length between embedded input and positional encoding. "
            f"Embedded input sequence length: {x.size(-2)}, "
            f"Positional encoding length: {self.positional_encoding.size(0)}"
)
        x = x + self.positional_encoding
        output = self.model(x)
        return output
    
    def summary(self):
        input_shape = (self.embedding_config['max_len'], 1)
        return summary(self, input_shape)

class SignalEmbedding(Module):

    def __init__(self, window_size:int, max_len:int, embed_dim:int=512, device=None, dtype=None) -> None:
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.embed_dim = embed_dim
        self.conv1d = Conv1d(in_channels=1,
                             out_channels=embed_dim,
                             kernel_size=window_size,
                             stride=window_size,
                             **factory_kwargs)
        self.expected_len = math.floor((max_len + 0 - window_size) / window_size) + 1
              
    def forward(self, input:Tensor) -> Tensor:
        is_batched = input.dim() == 3
        if is_batched:
            x = input.permute(0, 2, 1)
            x = self.conv1d(x)
            x = x.permute(0, 2, 1)
        else:
            x = input.permute(1, 0)
            x = self.conv1d(x)
            x = x.permute(1, 0)
        return x