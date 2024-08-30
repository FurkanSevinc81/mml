
import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np
import pandas as pd

def save_checkpoint(model, optimizer, epoch, name: str, path: str) -> str:
    """
    Saves the state of a PyTorch model to a checkpoint file.

    Saves the state dictionary of the given model to a  
    checkpoint file specified by path and name. The directory 
    is created if it doesn't already exist.

    Args:
        model: The PyTorch model to save.
        name (str): The name of the checkpoint file.
        path (str): The directory path where the checkpoint will be saved.

    Returns:
        str: The full path of the saved checkpoint file.

    Example:
        >>> model = MyPyTorchModel()
        >>> checkpoint_path = save_checkpoint(model, "model_v1.pth", "./checkpoints")
        >>> print(f"Checkpoint saved to: {checkpoint_path}")
    """
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, name)
    torch.save({'model_state_dict':model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch}, file_path)
    return file_path

def save_model(model, name: str, path: str):
    """
    Saves the entire PyTorch model to a file.

    Saves the complete model object, including its architecture 
    and parameters, to a file specified by path and name. The directory 
    is created if it doesn't already exist. his allows for direct loading 
    of the model without needing to instantiate the model class separately.

    Args:
        model: The PyTorch model to save.
        name (str): The name of the checkpoint file.
        path (str): The directory path where the checkpoint will be saved.

    Returns:
        str: The full path of the saved checkpoint file.

    Example:
        >>> model = MyPyTorchModel()
        >>> model_path = save_model(model, "full_model_v1.pth", "./models")
        >>> print(f"Model saved to: {model_path}")
    """
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, name)
    torch.save(model, file_path)
    return file_path

def load_from_checkpoint(model, optimizer, path) -> int:
    """
    Load a PyTorch model's state from a checkpoint file.

    Loads the state dictionary from the specified checkpoint file
    and applies it to the given model. It checks for the existence of the file
    before attempting to load it.

    Args:
        model (torch.nn.Module): The PyTorch model to load the state into.
            This should be an instance of the model with the correct architecture.
        optimizer: The optimzer to load the state into.
        path (str): The full path to the checkpoint file.

    Note:
        This function assumes that the checkpoint file contains a state dictionary
        that is compatible with the provided model's architecture. If the architectures
        don't match, a RuntimeError will be raised when attempting to load the state dict.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f'No checkpoint found at {path}')
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return epoch

def load_model(path: str):
    """
    Load an entire PyTorch model object from a file.

    This function loads a complete model object, including its architecture
    and parameters, from a file specified by the given path. It checks for
    the existence of the file before attempting to load it.

    Args:
        path (str): The full path to the saved model file.

    Returns:
        torch.nn.Module: The loaded PyTorch model.

    Example:
        >>> model_path = "./models/full_model_v1.pth"
        >>> loaded_model = load_model(model_path)
        >>> print(f"Model loaded successfully from: {model_path}")

    Note:
        This function assumes that the file contains a complete PyTorch model
        object saved using torch.save(model, path). It's designed to work with
        models saved using the save_model function. Be aware that loading models
        saved in this way can be sensitive to code changes and PyTorch versions.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f'No model file found at {path}')
    
    model = torch.load(path)
    
    if not isinstance(model, torch.nn.Module):
        raise ValueError("The loaded object is not a PyTorch model.")
    
    return model