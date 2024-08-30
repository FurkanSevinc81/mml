import torch
from torch.nn import Module, Conv1d, Parameter, Linear, Dropout, \
    ReLU, Sequential, AdaptiveAvgPool1d, LeakyReLU, GELU, ELU, BatchNorm1d
from torch.nn.init import xavier_normal_
from torch import Tensor
import torch.nn.functional as F
import math
from ..utils import kernel_functions as kops
from ..layers.kernel_transformer import KernelTransformer
from typing import Dict, Any
from .model_utils import summary, save_checkpoint, load_from_checkpoint


embed_config_basic = {
    'window_size': 10,
    'max_len': 2816,
}

embed_config_medium = {
    'window_size': 20,
    'max_len': 2816,
}

embed_config_high = {
    'window_size': 40,
    #'max_len': 2816,
}

"""
    In terms of size and parameter count this config 
    is the same as the base model described in the 
    original Transformer paper.
"""
kernel_transformer_config_base = {
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

kernel_transformer_config_dummy = {
    'kernel_function': kops.ExponentialKernel(),  # This should be set separately as it's a Callable
    'd_model': 512,
    'nhead': 2,
    'num_layers': 1,
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


kernel_transformer_config_small = {
    'kernel_function': None,  # This should be set separately as it's a Callable
    'd_model': 256,
    'nhead': 4,
    'num_layers': 4,
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
kernel_transformer_config_medium = {
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

class PositionalEncoding(Module):
    def __init__(self, embed_dim:int, max_seq_len:int = 2820, 
                 dropout:float = 0.1, scale:bool = True,
                 device=None, dtype=None) -> None:
        super().__init__()
        self.dropout = Dropout(dropout)
        self.scale = scale
        self.embed_dim = embed_dim
        position = torch.arange(0, max_seq_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
            
        pos_encoding = torch.zeros(max_seq_len, embed_dim, device=device, dtype=dtype)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        pos_encoding = pos_encoding.unsqueeze(0)
        self.register_buffer('PE', pos_encoding)

    def forward(self, x):
        if self.scale:
            x = x * math.sqrt(self.embed_dim)
        x = x + self.PE[:, :x.size(1), :]
        return self.dropout(x)

class KernelTransformerModel(Module):

    def __init__(self, transformer_config: Dict[str, Any], 
                 embed_config: Dict[str, Any], classification_config: Dict[str, Any],
                 use_bn:bool=False
                ):
        super().__init__()
        self.factory_kwargs = {'device': transformer_config['device'],
                               'dtype': transformer_config['dtype']}
        self.transformer_config = transformer_config
        self.embed_config = embed_config
        self.classification_config = classification_config

        self.use_bn = use_bn
        self.use_cls = True if self.classification_config['mode'] == 'cls' else False

        self.embeddings = self._create_embeddings()
        self.positional_encoding = self._create_positional_encodings()
        self.model = self._create_model()
        self.classification_head = self._create_classification_head()

        if use_bn:
            self.batch_norm = BatchNorm1d(self.transformer_config['d_model'], **self.factory_kwargs)
        if self.use_cls:
            self.cls_token = self._create_cls_token()

    def _create_embeddings(self):
        return SignalEmbedding(embed_dim=self.transformer_config['d_model'],
                               **self.factory_kwargs,
                               **self.embed_config)
                               
    def _create_positional_encodings(self):
        return PositionalEncoding(embed_dim=self.transformer_config['d_model'],
                                  **self.factory_kwargs)
    
    def _create_model(self):
        return KernelTransformer(**self.transformer_config)
    
    def _create_classification_head(self):
        """if not self.use_cls:
            return ClassificationHeadSeq(input_dim=self.transformer_config['d_model'],
                                         activation=activation, num_hidden_layers=0,
                                         hidden_dim=hidden_dim,**self.factory_kwargs)
        return ClassificationHeadCLS(input_dim=self.transformer_config['d_model'],
                                     hidden_dim=hidden_dim, activation=activation, 
                                     **self.factory_kwargs)"""
        return ClassificationHead(input_dim=self.transformer_config['d_model'],
                                  **self.classification_config,
                                  **self.factory_kwargs)
    
    def _create_cls_token(self):
        cls_token = Parameter(torch.empty(1, 1, self.transformer_config['d_model'], 
                                       **self.factory_kwargs))
        torch.nn.init.xavier_normal_(cls_token)
        return cls_token
    
    def forward(self, input: Tensor) -> Tensor:
        is_batched = input.dim() == 3
        if is_batched and not self.transformer_config['batch_first']:
            input = input.transpose(0, 1)

        x = self.embeddings(input)
        if self.use_cls:
            x = self._prepend_cls(x, is_batched)
        if self.use_bn:
            x = x.permute(0, 2, 1)
            x = self.batch_norm(x)
            x = x.permute(0, 2, 1)
        x = self.positional_encoding(x)
        x = self.model(x)

        if self.use_cls:
            cls_token = self._get_cls(x, is_batched)
            output = self.classification_head(cls_token)
        else:
            output = self.classification_head(x)
        return output
    
    def _prepend_cls(self, input:Tensor, is_batched:bool) -> Tensor:
        if is_batched:
            dim = 1
            cls_token = self.cls_token.expand(input.size(0), -1, -1)
        else:
            dim = 0
            cls_token = self.cls_token.squeeze(0)
        return torch.cat((cls_token, input), dim=dim)
    
    def _get_cls(self, input:int, is_batched:bool) -> Tensor:
        if is_batched:
            return input[:, 0, :]
        return input[0]

    def summary(self):
        input_shape = (self.embedding_config['max_len'], 1)
        return summary(self, input_shape)
    
    def save_checkpoint(self, name, path):
        return save_checkpoint(self, name, path)

    def load_checkpoint(self, path):
        load_from_checkpoint(self, path)


class KernelTransformerPretrain(Module):
    def __init__(self, transformer_config: Dict[str, Any], embed_config: Dict[str, Any]):
        super().__init__()
        self.factory_kwargs = {'device': transformer_config['device'],
                               'dtype': transformer_config['dtype']}
        self.transformer_config = transformer_config
        self.embed_config = embed_config
        self.embeddings = self._create_embeddings()
        self.positional_encoding = self._create_positional_encodings()
        self.model = self._create_model()

    def forward(self, input: Tensor, input_mask:None) -> Tensor:
        is_batched = input.dim() == 3
        if is_batched and not self.transformer_config['batch_first']:
            input = input.transpose(0, 1)
        x = self.embeddings(input)
        x = self.positional_encoding(x)

        output = self.model(x)
        return output
    
    def _create_embeddings(self):
        return SignalEmbedding(embed_dim=self.transformer_config['d_model'],
                               **self.factory_kwargs,
                               **self.embedding_config)

    def _create_positional_encodings(self):
        return PositionalEncoding(embed_dim=self.transformer_config['d_model'],
                                  **self.factory_kwargs)
    
    def _create_model(self):
        return KernelTransformer(**self.transformer_config)

class SignalEmbedding(Module):

    def __init__(self, window_size:int, embed_dim:int=512, device=None, dtype=None) -> None:
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.embed_dim = embed_dim
        self.conv1d = Conv1d(in_channels=1,
                             out_channels=embed_dim,
                             kernel_size=window_size,
                             stride=window_size,
                             **factory_kwargs)
        #self.expected_len = math.floor((max_len + 0 - window_size) / window_size) + 1
              
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

class ClassificationHead(Module):
    def __init__(self, mode:str='cls',
                 input_dim:int=512,
                 hidden_dim:int=None,
                 num_hidden_layers:int=0,
                 activation=None,
                 dropout:float=0.1,
                 device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.modes = {'cls', 'seq'}
        self.mode = mode
        if self.mode not in self.modes:
            raise ValueError(f'Unkown mode {mode}')
        if self.mode == 'seq':
            self.avg_pool = AdaptiveAvgPool1d(1)

        layers = []
        if hidden_dim is not None:
            current_dim = input_dim
            for _ in range(num_hidden_layers):
                layers.extend([
                    Linear(current_dim, hidden_dim),
                    activation,
                    Dropout(dropout)
                ])
                current_dim = hidden_dim
        layers.append(Linear(current_dim, 1))
        self.classifier = Sequential(*layers)
        self.to(**factory_kwargs)

    def forward(self, input:Tensor) -> Tensor:
        if self.mode == 'seq':
            x = self.avg_pool(input.transpose(1, 2))
            input = x.squeeze(-1)
        logits = self.classifier(input)
        return logits    


class ClassificationHeadCLS(Module):
    def __init__(self, input_dim:int=512, hidden_dim:int=None, 
                 activation=None, dropout:float=0.1, 
                 device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        layers = []

        if hidden_dim is not None:
            layers.extend([
                Linear(input_dim, hidden_dim),
                activation, #GELU(), #LeakyReLU(), # ReLU
                Dropout(dropout)
            ])
            input_dim = hidden_dim

        layers.append(Linear(input_dim, 1))

        # TODO maybe init with groot
        self.classifier = Sequential(*layers)
        self.to(**factory_kwargs)

    def forward(self, input:Tensor) -> Tensor:
        return self.classifier(input)

class ClassificationHeadSeq(Module):
    def __init__(self, input_dim:int=512, activation=None,
                 num_hidden_layers:int=0, hidden_dim:int=None, 
                 dropout:float=0.1, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.avg_pool = AdaptiveAvgPool1d(1)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        layers = []
        current_dim = input_dim
        if hidden_dim is not None:
            for _ in range(num_hidden_layers):
                layers.extend([
                    Linear(current_dim, hidden_dim),
                    activation, # ReLU
                    Dropout(dropout)
                ])
                current_dim = hidden_dim
        layers.append(Linear(current_dim, 1))
        self.classifier = Sequential(*layers)
        self.to(**factory_kwargs)

    def forward(self, input:Tensor) -> Tensor:
        x = self.avg_pool(input.transpose(1, 2))
        x = x.squeeze(-1)
        logits = self.classifier(x)
        return logits