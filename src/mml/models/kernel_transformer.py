import torch
from torch.nn import Module, Conv1d, Parameter, Linear, Dropout, ReLU, Sequential
from torch.nn.init import xavier_normal_
from torch import Tensor
import torch.nn.functional as F
import math
from ..utils import kernel_functions as kops
from ..layers.kernel_transformer import KernelTransformer
from typing import Dict, Any
from .model_utils import summary


embed_config_basic = {
    'window_size': 10,
    'max_len': 2816,
}

"""
    In terms of size and parameter count this config 
    is the same as the base model described in the 
    original Transformer paper.
"""
kernerl_transformer_config_base = {
    'kernel_function': kops.ExponentialKernel(),  # This should be set separately as it's a Callable
    'd_model': 512,
    'nhead': 8,
    'num_layers': 6,
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


kernerl_transformer_config_small = {
    'kernel_function': None,  # This should be set separately as it's a Callable
    'd_model': 256,
    'nhead': 1,#4,
    'num_layers': 1,#4,
    'dim_feedforward': 1024,
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

"""
    In terms of size and parameter count this config 
    is the same as the base BERT and ViT model.
"""
kernerl_transformer_config_medium = {
    'kernel_function': None,  # This should be set separately as it's a Callable
    'd_model': 768,
    'nhead': 12,
    'num_layers': 12,
    'dim_feedforward': 3072,
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

"""
    In terms of size and parameter count this config 
    is the same as the base BERT and ViT model.
"""
kernerl_transformer_config_large= {
    'kernel_function': None,  # This should be set separately as it's a Callable
    'd_model': 1024,
    'nhead': 16,
    'num_layers': 24,
    'dim_feedforward': 4096,
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


def positional_encoding_sin_cos(embed_dim:int, max_seq_len:int, device=None, dtype=None) -> Tensor:
        # TODO select device
        position = torch.arange(max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        pos_encoding = torch.zeros(max_seq_len, embed_dim, device=device, dtype=dtype)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding

class KernelTransformerModel(Module):

    def __init__(self, use_cls: bool, transformer_config: Dict[str, Any], embed_config: Dict[str, Any]):
        super().__init__()
        self.factory_kwargs = {'device': transformer_config['device'],
                               'dtype': transformer_config['dtype']}
        self.transformer_config = transformer_config
        self.embedding_config = embed_config
        self.use_cls = use_cls
        if self.use_cls:
            self.embedding_config['max_len'] += 1
            self.cls_token = Parameter(torch.empty(1, 1, self.transformer_config['d_model'], 
                                       **self.factory_kwargs))
            torch.nn.init.xavier_normal_(self.cls_token)
            self.classification_head = self._create_classification_head()
        self.embeddings = self._create_embeddings()
        self.positional_encoding = self._create_positional_encodings()
        self.model = self._create_model()

    def _create_embeddings(self):
        return SignalEmbedding(embed_dim=self.transformer_config['d_model'],
                               **self.factory_kwargs,
                               **self.embedding_config)
                               
    def _create_positional_encodings(self):
        c = 1 if self.use_cls else 0
        return positional_encoding_sin_cos(embed_dim=self.transformer_config['d_model'],
                                           max_seq_len=self.embeddings.expected_len+c,
                                           **self.factory_kwargs
                                           )
    def _create_model(self):
        return KernelTransformer(**self.transformer_config)
    
    def _create_classification_head(self):
        if not self.use_cls:
            return None
        return BinaryClassificationHead(input_dim=self.transformer_config['d_model'],
                                        hidden_dim=2,
                                        **self.factory_kwargs)
    
    def forward(self, input: Tensor) -> Tensor:
        x = self.embeddings(input)
        is_batched = input.dim() == 3
        if self.transformer_config['batch_first'] and is_batched:
            B = x.size(0)
        elif not self.transformer_config['batch_first'] and is_batched:
            B = x.size(1)

        if self.use_cls: 
            x = self._prepend_cls(x, B, is_batched)
        x = x + self.positional_encoding
        x = self.model(x)
        if self.use_cls:
            cls_token = self._get_cls(x, is_batched)
            output = self.classification_head(cls_token)
        return output
    
    def _prepend_cls(self, input:Tensor, batch:int, is_batched:bool) -> Tensor:
        if is_batched:
            x, y, dim = (batch, -1, 1) if self.transformer_config['batch_first'] else (-1, batch, 0)
            cls_token = self.cls_token.expand(x, y, -1)
        else:
            dim = 0
            cls_token = self.cls_token.unsqueeze(0)
        #cls_token = cls_token.requires_grad()
        return torch.cat((cls_token, input), dim=dim)
    
    def _get_cls(self, input:int, is_batched:bool) -> Tensor:
        if is_batched:
            cls_token = input[:, 0, :] if self.transformer_config['batch_first'] else input[0, :, :]
            return cls_token
        return input[0, :]

    def summary(self):
        input_shape = (self.embedding_config['max_len'], 1)
        return summary(self, input_shape)

class SignalEmbedding(Module):

    def __init__(self, window_size:int, max_len:int, embed_dim:int=512, device=None, dtype=None) -> None:
        # TODO device dtype
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

class BinaryClassificationHead(Module):
    def __init__(self, input_dim:int=512, hidden_dim:int=None, 
                 dropout_rate:float=0.1, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        layers = []

        if hidden_dim is not None:
            layers.extend([
                Linear(input_dim, hidden_dim),
                ReLU(),
                Dropout(dropout_rate)
            ])
            input_dim = hidden_dim

        layers.append(Linear(input_dim, 1))

        self.classifier = Sequential(*layers)
        self.to(**factory_kwargs)

    def forward(self, input:Tensor) -> Tensor:
        return self.classifier(input)



class ClassificationHead(Module):
    def __init__(self, input_dim:int=512, num_classes:int=2, hidden_dim:int=None, dropout_rate:float=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        layers = []

        if hidden_dim is not None:
            layers.extend([
                Linear(input_dim, hidden_dim),
                ReLU(),
                Dropout(dropout_rate)
            ])
            input_dim = hidden_dim

        layers.append(Linear(input_dim, num_classes))

        self.classifier = Sequential(*layers)

    def forward(self, input:Tensor) -> Tensor:
        return self.classifier(input)
        """if self.num_classes == 2:
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=-1)"""