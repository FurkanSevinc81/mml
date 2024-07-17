"""
    source: https://github.com/sksq96/pytorch-summary
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np
import seaborn as sns
import pandas as pd

def save_checkpoint(model, name, path):
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, name)
    torch.save(model.state_dict(), file_path)
    return file_path

def load_from_checkpoint(model, path):
    if not os.path.exists(path):
        raise FileNotFoundError(f'No checkpoint found at {path}')
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    return model

def analyze_cls_token(cls_list):
    # Convert list of tensors to a numpy array
    cls_array = torch.stack([tensor.detach().cpu() for tensor in cls_list]).squeeze().numpy()
    
    # Get the number of training steps and embedding dimension
    num_steps, embed_dim = cls_array.shape
    
    # Create a heatmap
    plt.figure(figsize=(12, 8))
    plt.imshow(cls_array.T, aspect='auto', interpolation='nearest')
    plt.colorbar(label='Value')
    plt.title('CLS Token Values Over Training')
    plt.xlabel('Training Step')
    plt.ylabel('Embedding Dimension')
    plt.show()
    
    # Plot mean and std dev over time
    mean = np.mean(cls_array, axis=1)
    std = np.std(cls_array, axis=1)
    
    plt.figure(figsize=(12, 6))
    plt.plot(mean, label='Mean')
    plt.fill_between(range(num_steps), mean - std, mean + std, alpha=0.3, label='Std Dev')
    plt.title('CLS Token Statistics Over Training')
    plt.xlabel('Training Step')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
    
    # Plot a few individual dimensions
    plt.figure(figsize=(12, 6))
    for i in range(min(5, embed_dim)):  # Plot first 5 dimensions or fewer
        plt.plot(cls_array[:, i], label=f'Dim {i}')
    plt.title('Sample of Individual CLS Token Dimensions')
    plt.xlabel('Training Step')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

def plot_gradient_norms(norms, save_path=None):
    plt.figure(figsize=(12, 6))
    plt.plot(norms, marker='o')
    plt.title('Gradient Norm Over Training Steps')
    plt.xlabel('Training Step')
    plt.ylabel('Gradient Norm')
    plt.yscale('log')  # Log scale is often useful for gradient norms
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Add horizontal lines for reference
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=0.1, color='g', linestyle='--', alpha=0.5)
    plt.axhline(y=10, color='y', linestyle='--', alpha=0.5)
    
    plt.legend(['Gradient Norm', 'y=1', 'y=0.1', 'y=10'])
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def get_layer_wise_gradients(model):
    layer_grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            layer_grads[name] = param.grad.abs().mean().item()
    return layer_grads

def plot_layer_wise_gradients(all_layer_grads, save_path=None, display_df=True):
    df = pd.DataFrame(all_layer_grads)
    
    if display_df:
        print("Layer-wise Gradient Statistics:")
        print(df.describe().to_string())
        
        print("\nFirst few rows of the gradient data:")
        print(df.head().to_string())
        
        print("\nLast few rows of the gradient data:")
        print(df.tail().to_string())
    
    plt.figure(figsize=(20, 15))

    ax = sns.heatmap(df.T, cmap='viridis', cbar_kws={'label': 'Gradient Magnitude'})
    
    plt.title('Layer-wise Gradient Magnitudes Over Training Steps')
    plt.xlabel('Training Step')
    plt.ylabel('Layer')
    
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right')
    plt.tight_layout()  
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

    plt.figure(figsize=(15, 7))
    selected_layers = df.columns[:10] 
    for layer in selected_layers:
        plt.plot(df[layer], label=layer)
    
    plt.title('Gradient Magnitudes for Selected Layers')
    plt.xlabel('Training Step')
    plt.ylabel('Gradient Magnitude')
    plt.yscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()  
    
    if save_path:
        plt.savefig(save_path.replace('.png', '_selected_layers.png'), bbox_inches='tight', dpi=300)
    plt.show()

def plot_training_progress(epochs, train_losses, train_accuracies, val_losses, val_accuracies, title):
    """
    Plots training and validation losses and accuracies over epochs with a given title.

    Args:
        epochs (int): Number of epochs
        train_losses (list): List of training losses
        train_accuracies (list): List of training accuracies
        val_losses (list): List of validation losses
        val_accuracies (list): List of validation accuracies
        title (str): Title for the plots
    """
    epochs_range = range(1, epochs + 1)

    plt.figure(figsize=(14, 7))
    
    # Main title for the figure
    plt.suptitle(title, fontsize=16)

    # Plotting training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Training Loss', marker='o')
    plt.plot(epochs_range, val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plotting training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label='Training Accuracy', marker='o')
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def summary(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
    if device.type == 'cuda' and not torch.cuda.is_available():
        print('Warning: CUDA is not available. Falling back to CPU.')
        device = torch.device('cpu')
    result, params_info = summary_string(
        model, input_size, batch_size, device, dtypes)
    print(result)

    return params_info

def summary_string(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
    if dtypes == None:
        dtypes = [torch.FloatTensor]*len(input_size)

    summary_str = ''

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
        ):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype).to(device=device)
         for in_size, dtype in zip(input_size, dtypes)]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    summary_str += "----------------------------------------------------------------" + "\n"
    line_new = "{:>20}  {:>25} {:>15}".format(
        "Layer (type)", "Output Shape", "Param #")
    summary_str += line_new + "\n"
    summary_str += "================================================================" + "\n"
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]

        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        summary_str += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(sum(input_size, ()))
                           * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. /
                            (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    summary_str += "================================================================" + "\n"
    summary_str += "Total params: {0:,}".format(total_params) + "\n"
    summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    summary_str += "Non-trainable params: {0:,}".format(total_params -
                                                        trainable_params) + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
    summary_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
    summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
    summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    # return summary
    return summary_str, (total_params, trainable_params)