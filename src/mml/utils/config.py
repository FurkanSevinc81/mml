import yaml
import os
from functools import reduce
from operator import getitem
from argparse import ArgumentParser
import mml
import torch
from datetime import datetime
import torch.nn as nn
import mml.utils.kernel_functions as kf
from mml.models.kernel_transformer import KernelTransformerModel
from mml.trainer import Trainer
from mml.data.biovid import train_test_dataloader
from mml.utils.logger import Logger
import copy
import torch.distributed as dist

model_options = ['base', 'small', 'medium', 'large']

embedding_options = ['basic', 'medium', 'high']

activation_map = {
    "relu": nn.ReLU(),
    "leaky_relu": nn.LeakyReLU(),
    "elu": nn.ELU(),
    "selu": nn.SELU(),
    "gelu": nn.GELU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "softmax": nn.Softmax(dim=-1),
    "softplus": nn.Softplus(),
    "softsign": nn.Softsign(),
    "relu6": nn.ReLU6(),
    "prelu": nn.PReLU(),
    "swish": nn.SiLU(),  
    "logsigmoid": nn.LogSigmoid(),
    "hardtanh": nn.Hardtanh(),
    "hardswish": nn.Hardswish(),
    "hardsigmoid": nn.Hardsigmoid(),
    "tanhshrink": nn.Tanhshrink(),
    "threshold": nn.Threshold(0.1, 0),
}

kernel_function_map = {
    'linear': lambda **kwargs: kf.LinearKernel(**kwargs),
    'polynomial': lambda **kwargs: kf.PolynomialKernel(**kwargs),
    'rbf': lambda **kwargs: kf.RBFKernel(**kwargs),
    'sigmoid': lambda **kwargs: kf.SigmoidKernel(**kwargs),
    'laplacian': lambda **kwargs: kf.LaplacianKernel(**kwargs),
    'exponential': lambda **kwargs: kf.ExponentialKernel(**kwargs),
}

class ConfigParser():
    def __init__(self, config=None, modification=None, resume=False):
        # TODO add support for multi GPU
        #n_gpus = torch.cuda.device_count()
        #self.ddp = False
        #if n_gpus >= 2:
        #   self.ddp = True
        #   self._setup_ddp()
        #else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resume = resume
        self.config = self._parse_config(config)
        self.config = _update_config(self.config, modification)
        self.factory_kwargs = {'device': self.config['factory_kwargs']['device'], 
                               'dtype': self.config['factory_kwargs']['dtype']}
        self.factory_kwargs['device'] = device
        self.predefined_conf_dir = self.config['predefined']
        
        save_dir = self.config['save_dir'] # check if not None else error
        expr_name = self.config['name']
        run_id = self.config['run_id']
        if self.config['resume']['expr_path'] is not None and resume:
            # check if path is correct/exists
            expr_path = self.config['resume']['expr_path']
        else:      
            if run_id is None:
                run_id = datetime.now().strftime(r'%m%d_%H%M%S')
            expr_path = os.path.join(save_dir, expr_name, run_id)
            self.config['run_id'] = run_id
            self.config['resume']['expr_path'] = expr_path

        self.model_save_dir = os.path.join(expr_path, 'models')
        self.log_save_dir = os.path.join(expr_path, 'logs')
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.log_save_dir, exist_ok=True)

        self.logger = Logger(log_config=self.config['log_config'],
                             verbosity=self.config['verbosity'],
                             save_dir_model=self.model_save_dir,
                             save_dir_logs=self.log_save_dir)
        #### LOGGING  ####
        self.logger.log(f'Starting experiment: {expr_name} with ID {run_id}',
                        verbosity=1)
        self.logger.log(f"Using log config: {self.config['log_config']}", verbosity=1)
        self.logger.log(f'Log directory:\t{self.log_save_dir}', verbosity=1)
        self.logger.log(f'Model save directory:\t{self.model_save_dir}', verbosity=1)
        self.logger.log(f'Using settings from:\t{config}', verbosity=1)
        self.saved_conf = os.path.join(self.model_save_dir, 'config.yaml')
        self.logger.log(f'Saving new config to:\t{self.saved_conf}', verbosity=1)
        self.logger.save_config(self.config, self.saved_conf)

    def _parse_config(self, config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config
    
    def _setup_ddp(self, rank, world_size):
        backend = 'gloo' if os.name == 'nt' else 'nccl'  
        os.environ.setdefault('MASTER_ADDR', 'localhost')
        os.environ.setdefault('MASTER_PORT', '12355')
        dist.init_process_group(backend=backend, rank=rank,
                                world_size=world_size)
    def _cleanup_ddp(self):
        dist.destroy_process_group()

    def init_dataloader(self):
        self.logger.log(f'Initializing train and test dataloader', verbosity=2)
        data_conf = copy.deepcopy(self.config['data'])
        if data_conf['transform'] is not None:
            transform = getattr(mml.data.biovid, data_conf['transform'])
            data_conf['transform'] = transform()
        train_dl, test_dl, train_id, test_id = train_test_dataloader(**data_conf)
        
        self.config['data']['test_ids'] = test_id
        self.config['data']['train_ids'] = train_id
        #### LOGGING  ####
        self.logger.log(f'Subjects used for training: {train_id}', verbosity=1)
        self.logger.log(f'Subjects used for testing: {test_id}', verbosity=1)
        self.logger.log(f'Saving test and train IDs to:\t{self.saved_conf}', verbosity=1)
        self.logger.save_config(self.config, self.saved_conf)

        return train_dl, test_dl
    
    def init_model(self):
        self.logger.log(f'Initializing kernel transformer model', verbosity=2)
        model_conf = self.config['model']
        if self.config['load_model'] is not None:
            # load a KernelTransformer object from file and train with provided config
            self.logger.log(f'Loading model object from {self.config["load_model"]}')
            model = KernelTransformerModel.load_model(self.config['load_model'])
            return model

        transformer_conf= model_conf['kernel_transformer']
        embedding_conf = model_conf['embedding']
        classification_conf = model_conf['classification']

        pre = self._parse_config(os.path.join(self.predefined_conf_dir, 'model.yaml'))

        if transformer_conf in model_options:
            self.logger.log(f"Found predefined model. Loading '{transformer_conf}'", verbosity=2)
            transformer_conf = pre['kernel_transformer'][transformer_conf]
        else:
            self.logger.log(f'Found custom model. Loading model config from {transformer_conf}', verbosity=2)
            transformer_conf = self._parse_config(transformer_conf)

        if embedding_conf in embedding_options:
            embedding_conf = pre['embedding'][embedding_conf]
        else:
            embedding_conf = self._parse_config(embedding_conf)

        transformer_conf['activation'] = activation_map[transformer_conf['activation']]
        
        kernel_name= model_conf['kernel_function']['name']
        kernel_kwargs = model_conf['kernel_function']['kwargs']
        if kernel_kwargs:
            kernel_function = kernel_function_map[kernel_name](**self.factory_kwargs, **kernel_kwargs)
        else:
            kernel_function = kernel_function_map[kernel_name](**self.factory_kwargs)
        
        transformer_conf['kernel_function'] = kernel_function
        classification_conf['activation'] = activation_map[classification_conf['activation']]

        model_conf = {
            'transformer_config': transformer_conf,
            'embed_config': embedding_conf,
            'classification_config': classification_conf,
            'use_bn': model_conf['use_bn'],
            **self.factory_kwargs
        }
        model = KernelTransformerModel(**model_conf)
        return model

    def init_trainer(self):
        model = self.init_model()
        train_dl, val_dl = self.init_dataloader()
        self.config['optimizer']['kwargs']['params'] = model.parameters()
        optimizer = self._get_function(self.config['optimizer'], torch.optim)
        epoch_start = 0
        if self.resume and self.config['resume']['checkpoint'] is not None:
            assert os.path.exists(self.config['resume']['checkpoint']), \
            self.logger.log(f"No model file found in {self.config['resume']['checkpoint']}",
                             verbosity=0)   
            epoch_start = model.resume_checkpoint(optimizer, path=self.config['resume']['checkpoint'])
        # TODO check if multiple gpus are available. if so init parallel model
        #if self.ddp:
        #   model = DDP(model)
        # TODO if multi GPU dataloader should use sampler
        
        criterion = self._get_function(self.config['criterion'],
                                       torch.nn)

        classes = self.config['data']['classes']
        labels = [0, 1] if len(classes) == 2 else classes
        metrics_train_kwargs = self.config['metrics']['train']
        metrics_val_kwargs = self.config['metrics']['val']
        train_key_paths = self._get_key_path(metrics_train_kwargs, 'labels')
        val_key_paths = self._get_key_path(metrics_val_kwargs, 'labels')
        train_modification = {path: labels for path in train_key_paths}
        val_modification = {path: labels for path in val_key_paths}
        metrics_train_kwargs = _update_config(metrics_train_kwargs, train_modification)
        metrics_val_kwargs = _update_config(metrics_val_kwargs, val_modification)

        self.logger.log(f'Initializing Trainer', verbosity=2)
        return Trainer(model=model, criterion=criterion, optimizer=optimizer, 
                       train_dataloader=train_dl, val_dataloader=val_dl, 
                       kwargs_metrics_train= metrics_train_kwargs, 
                       kwargs_metrics_val=metrics_val_kwargs, epoch_start=epoch_start,
                       classes=self.config['data']['classes'],
                       logger=self.logger, **self.config['trainer'],
                       **self.factory_kwargs)
    
    @classmethod
    def from_args(cls, args: ArgumentParser, options=''):
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type, help=opt.help,
                              metavar=opt.metavar)
        if not isinstance(args, tuple):
            args = args.parse_args()
        if args.resume:
            resume_config = args.resume
            return  cls(resume_config, resume=True)

        modification = {opt.target : getattr(args, _get_opt_name(opt.flags)) 
                        for opt in options if getattr(args, _get_opt_name(opt.flags)) is not None}
        return cls(args.config, modification)

    def _get_function(self, config, module):
        function = getattr(module, config['name'])
        if config['kwargs']:
            return function(**config['kwargs'])
        return function()
    
    def _get_key_path(self, config, key, section=None):
        paths = []
        def recurse(current_dict, current_path):
            if isinstance(current_dict, dict):
                for k, v in current_dict.items():
                    new_path = f"{current_path};{k}"if current_path else k
                    if k == key:
                        paths.append(new_path)
                    elif isinstance(v, dict):
                        recurse(v, new_path)
        if section:
            if section in config:
                recurse(config[section], section)
            else:
                raise ValueError(f'Section {section} not found in config')
        else:
            recurse(config, '')
        return paths


def _update_config(config, modification):
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config

def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value

def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)

def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')