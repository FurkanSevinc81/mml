from typing import Optional, Tuple, Callable
import math
import torch
from torch import Tensor
import warnings

def _correct_dim(tensor:Tensor)->Tensor:
    if tensor.dim() <= 1:
        return tensor.transpose(-1, 0)
    else:
        return tensor.transpose(-2, -1)
    
def _to_float(tensor:Tensor, precision:str='float32')->Tensor:
    """
    Converts a given PyTorch tensor to a float tensor with the specified precision.
    
    Args:
        tensor (torch.Tensor): The input tensor to be converted.
        precision (str): The precision of the output tensor. Can be 'float16', 
                        'float32' or 'float64'. Default is 'float32'.
    
    Returns:
        torch.Tensor: The tensor converted to the specified float precision.
    
    Raises:
        ValueError: If the specified precision is not supported.
    """
    if precision == 'float16':
        if tensor.is_cuda:
            return tensor.to(torch.float16)
        else:
            warnings.warn(f"{precision} is not supported on CPU; using float32 instead")
            return tensor.to(torch.float32)
    elif precision == 'float32':
        return tensor.to(torch.float32)
    elif precision == 'float64':
        return tensor.to(torch.float64)
    else:
        raise ValueError(f"Unsupported precision: {precision}."\
                          "Supported precisions are 'float16', 'float32' and 'float64'.")

def masked_kernel(kernel_func: Callable[[Tensor, Tensor], Tensor], 
                  x: Tensor, 
                  y: Tensor,
                  mask: Tensor = None, 
                  is_causal: bool = False) -> Tensor:
    """
    Computes the masked kernel matrix for the given inputs and mask using an arbitrary kernel function.

    Args:
      kernel_func: The kernel function.
      x: The first input tensor with shape (m, d).
      y: The second input tensor with shape (n, d).
      mask: The mask tensor with shape (m, n).

    Returns:
      torch: The masked kernel matrix with shape (m, n).
    """
    # TODO
    if is_causal:
        assert mask is None
        L, S = x.size(-2), y.size(-2)
        mask = create_causal_mask(L, S, x.dtype)
    kernel = kernel_func(x, y)
    kernel += mask
    return kernel
    

def kernel_smoother(kernel: Tensor) -> Tensor:
    # TODO
    sum = torch.sum(kernel, -1, keepdim=True)
    return kernel / sum

def create_causal_mask(L, S, dtype):
    """
    Args:
        L: Target sequence length
        S: Source sequence length
    
    """
    mask = torch.zeros(L, S, dtype=dtype)
    temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
    mask.masked_fill_(temp_mask.logical_not(), float("-inf"))
    mask.to(dtype)
    return mask

def kernel_based_attention(query: Tensor, 
                           key: Tensor, 
                           value:Tensor, 
                           kernel_function: Callable[[Tensor, Tensor], Tensor], 
                           attn_mask: Optional[Tensor] = None,
                           is_causal: bool = False, 
                           dropout_p = 0.0) -> Tensor:
    # TODO
    attn_weight = masked_kernel(kernel_function, query, key, attn_mask, is_causal)
    attn_weight = kernel_smoother(attn_weight)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    value = _correct_dim(value)
    return torch.matmul(attn_weight, value)


def multi_head_attention_forward( kernel_func: Callable[[Tensor, Tensor], Tensor],
        query: Tensor,
        key: Tensor,
        value: Tensor,

        num_heads: int,
        in_proj_weight: Optional[Tensor],
        in_proj_bias: Optional[Tensor],
        bias_k: Optional[Tensor],
        bias_v: Optional[Tensor],

        dropout_p: float,
        out_proj_weight: Tensor,
        out_proj_bias: Optional[Tensor],
        training: bool = True,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        use_separate_proj_weight: bool = False,
        q_proj_weight: Optional[Tensor] = None,
        k_proj_weight: Optional[Tensor] = None,
        v_proj_weight: Optional[Tensor] = None,
        static_k: Optional[Tensor] = None,
        static_v: Optional[Tensor] = None,
        average_attn_weights: bool = True,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Missing:
        q, k
    """
    if need_weights:
        # these are not required since the kernel function operates directly on Q and K:
        # A = k(Q, A)
        #B, Nt, E = q.shape # assuming input has shapes (batch, sample, embedding)
        #q_scaled = q / math.sqrt(E) # calculate Q' = Q / sqrt(d_k) 
        if attn_mask is not None: # calculate attention weights A = k(Q, A)
            attn_ouput_weights = masked_kernel(kernel_func, attn_mask, q, k)
        else:
            attn_ouput_weights = kernel_func(q, k)
        attn_ouput_weights = kernel_smoother(attn_ouput_weights)

