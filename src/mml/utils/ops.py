from typing import Optional, Tuple, Callable
import math
import torch
from torch import Tensor
import torch.nn.functional as F
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

def calculate_kernel(kernel_func: Callable[[Tensor, Tensor], Tensor],
                     x: Tensor,
                     y: Tensor,
                     mask: Tensor = None,
                     is_causal: bool = False) -> Tensor:
    """
    Computes the (masked) kernel matrix for the given inputs and mask using an arbitrary kernel function.
    This is essentially used for the calculation of the attention scores between queries Q and keys K, where:
        A = k(Q, K)
    The kernel function here replaces the inner product used in the scaled dot-product attention of the original 
    Transformer. This function assumes the data to be batch first.

    Args:
      kernel_func: The kernel function.
      x: The first input tensor with shape (B, m, d), where B is the batch dimension
      y: The second input tensor with shape (B, n, d), where B is the batch dimension
      mask: The mask tensor with shape (m, n).

    Returns:
      torch: The masked kernel matrix with shape (m, n).
    """

    if is_causal:
        assert mask is None
        L, S = x.size(-2), y.size(-2)
        mask = create_causal_mask(L, S, x.dtype)
    if kernel_func.has_stable:
        kernel = kernel_func(x, y, stable=True)
        if mask is not None:
            kernel += mask
    else:
        kernel = kernel_func(x, y)
        # while scaled dot-product attention masks the inputs with negative infinity
        # the use of kernel functions requires a masking with 0.0. This achieves the
        # same results, since exp(-inf) evaluates to 0.0, such that the softmax of masked
        # input ensures that the masked positions are 0.0.
        if mask is not None:
            kernel = kernel.masked_fill(mask == float("-inf"), 0.0)

    return kernel

def kernel_smoother(kernel: Tensor) -> Tensor:
    """
    Smooths the given kernel matrix by normalizing it along the last dimension, 
    such that each row of the matrix adds up to 1.0. This is used to calculate 
    the kernel weighted average of values V, similiar to the softmax in scaled 
    dot-product attenion.

    Args:
        kernel (Tensor): A tensor representing the kernel, where normalization 
                         should be performed along the last dimension.

    Returns:
        Tensor: A tensor where each element in the last dimension is divided 
                by the sum of the elements in that dimension, resulting in 
                a normalized kernel.

    Example:
        # If kernel has shape (batch_size, num_features), this function 
        # normalizes each row in the last dimension.
        kernel = torch.tensor([[1.0, 2.0, 3.0], 
                               [4.0, 5.0, 6.0]])
        smoothed_kernel = kernel_smoother(kernel)
        # smoothed_kernel will be:
        # tensor([[0.1667, 0.3333, 0.5000],
        #         [0.2667, 0.3333, 0.4000]])
    """
    sum = torch.sum(kernel, -1, keepdim=True)
    return kernel / sum

def create_causal_mask(L: int, S: int, dtype: torch.dtype=torch.float32) -> Tensor:
    """
    Creates a causal mask to be used in attention mechanisms to prevent
    attending to future tokens.

    Args:
        L (int): Target sequence length.
        S (int): Source sequence length.
        dtype (torch.dtype): The data type for the mask tensor. Defaults to
                             `torch.float32`

    Returns:
        torch.Tensor: A causal mask tensor of shape (L, S) where elements
                      above the main diagonal are set to negative infinity
                      (-inf) and elements on or below the diagonal are set to 0.

    Example:
        # If L = 5 and S = 5, the returned mask will look like:
        # tensor([[  0., -inf, -inf, -inf, -inf],
        #         [  0.,   0., -inf, -inf, -inf],
        #         [  0.,   0.,   0., -inf, -inf],
        #         [  0.,   0.,   0.,   0., -inf],
        #         [  0.,   0.,   0.,   0.,   0.]], dtype=torch.float32)
    """
    #TODO Transformer module method `generate_square_subsequent_mask` provides similar functionality
    mask = torch.zeros(L, S, dtype=dtype)
    temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
    mask.masked_fill_(temp_mask.logical_not(), float("-inf"))
    return mask.to(dtype)

def kernel_based_attention(query: Tensor, 
                           key: Tensor, 
                           value:Tensor, 
                           kernel_function: Callable[[Tensor, Tensor], Tensor], 
                           attn_mask: Optional[Tensor] = None,
                           is_causal: bool = False, 
                           dropout_p = 0.0, 
                           need_weights = False,
                           batch_first = True
                           ) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Computes kernel-based attention using a specified kernel function. Input has to be batch first.

    Args:
        query (Tensor): The query tensor of shape (batch_size, num_heads, L, d_k) or (batch_size, L, d_k).
        key (Tensor): The key tensor of shape (batch_size, num_heads, S, d_k) or (batch_size, S, d_k).
        value (Tensor): The value tensor of shape (batch_size, num_heads, S, d_v) or (batch_size, S, d_v).
        kernel_function (Callable[[Tensor, Tensor], Tensor]): A function to compute the kernel, 
                                                              given query and key tensors.
        attn_mask (Optional[Tensor]): An optional mask tensor of shape (batch_size, num_heads, L, S)
                                      to apply to the attention weights.
        is_causal (bool): If True, applies a causal mask to prevent attending to future tokens.
        dropout_p (float): Dropout probability to apply to the attention weights.
        batch_first (bool): if False, query, key, and value are reshaped to be batch first. 
                            Defaults to True

    Returns:
        attn_output: The result of the attention mechanism applied to the value tensor,
                     with shape (batch_size, num_heads, L, d_v) or (batch_size, L, d_v).
        attn_weights: Only returned when `need_weights=True`.
        
    """
    if not batch_first:
        query, key, value = (x.transpose(1, 0) for x in (query, key, value))

    if kernel_function.has_stable:
        # Calculate the attention weights using the provided kernel function.
        attn_weights = calculate_kernel(kernel_function, query, key, attn_mask, is_causal)
        # having has_stable = True indicates that the kernel has exp(.) as outermost expression.
        # since calculating the kernel with stable set as True only returns the results inside
        # of exp, the regular softmax can be applied. Example RBF kernel:
        # RBF(x, y) = exp(-gamma * ||x-y||^2) = exp(z), where z = -gamma * ||x-y||^2
        # kernel_smoother(exp(-gamma * ||x-y||^2)) = softmax(-gamma * ||x-y||^2)
        attn_weights = F.softmax(attn_weights, dim=-1)
    else:
        attn_weights = calculate_kernel(kernel_function, query, key, attn_mask, is_causal)
        attn_weights = kernel_smoother(attn_weights)
    
    # Apply dropout to the attention weights if dropout_p is greater than 0.
    if dropout_p > 0.0:
        attn_weights = torch.dropout(attn_weights, dropout_p, train=True)
    # Compute the final attention output by performing a weighted sum of the value tensor.
    attn_output = torch.matmul(attn_weights, value)
    if need_weights:
        return attn_output, attn_weights
    return attn_output, None

def kernel_multi_head_attention_forward( 
        kernel_func: Callable[[Tensor, Tensor], Tensor],
        query: Tensor,
        key: Tensor,
        value: Tensor,
        embed_dim_to_check: int,
        num_heads: int,
        in_proj_weight: Optional[Tensor],
        in_proj_bias: Optional[Tensor],
        bias_k: Optional[Tensor],
        bias_v: Optional[Tensor],
        add_zero_attn: bool,
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
        is_causal: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Args:
        kernel_func: 
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
            Default: `True`
            Note: `needs_weight` defaults to `True`, but should be set to `False`
            For best performance when attention weights are not needed.
            *Setting needs_weights to `True`
            leads to a significant performance degradation.*
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        is_causal: If specified, applies a causal mask as attention mask, and ignores
            attn_mask for computing scaled dot product attention.
            Default: ``False``.
            .. warning::
                is_causal provides a hint that the attn_mask is the
                causal mask.Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across heads.
            Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an effect
            when ``need_weights=True.``. Default: True


    Shape:
        Inputs:
        - query: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, E)` or :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, E)` or :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(S)` or :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a FloatTensor is provided, it will be directly added to the value.
          If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: Only returned when ``need_weights=True``. If ``average_attn_weights=True``, returns
          attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
          head of shape :math:`(num_heads, L, S)` when input is unbatched or :math:`(N, num_heads, L, S)`.
    """
    ########################################################
    # Starting from here: the code below is identical to
    # `torch.nn.functional._multi_head_attention_forward`
    ########################################################
    is_batched = F._mha_shape_check(query, 
                                    key, 
                                    value, 
                                    key_padding_mask,
                                    attn_mask,
                                    num_heads)
    # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
    # is batched, run the computation and before returning squeeze the
    # batch dimension so that the output doesn't carry this temporary batch dimension.
    if not is_batched:
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape

    key_padding_mask = F._canonical_mask(
        mask=key_padding_mask,
        mask_name="key_padding_mask",
        other_type=F._none_or_dtype(attn_mask),
        other_name="attn_mask",
        target_type=query.dtype
    )

    if is_causal and attn_mask is None:
        raise RuntimeError(
            "Need attn_mask if specifying the is_causal hint. "
            "You may use the Transformer module method "
            "`generate_square_subsequent_mask` to create this mask."
        )

    if is_causal and key_padding_mask is None and not need_weights:
        # when we have a kpm or need weights, we need attn_mask
        # Otherwise, we use the is_causal hint go as is_causal
        # indicator to SDPA.
        attn_mask = None
    else:
        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        if key_padding_mask is not None:
            # We have the attn_mask, and use that to merge kpm into it.
            # Turn off use of is_causal hint, as the merged mask is no
            # longer causal.
            is_causal = False

    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert key.shape[:2] == value.shape[:2], \
            f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        assert in_proj_weight is not None, "use_separate_proj_weight is False but in_proj_weight is None"
        q, k, v = F._in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
        assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
        assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = F._in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make them batch first
    #
    q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        assert static_k.size(0) == bsz * num_heads, \
            f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        assert static_k.size(2) == head_dim, \
            f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        k = static_k
    if static_v is None:
        v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        assert static_v.size(0) == bsz * num_heads, \
            f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        assert static_v.size(2) == head_dim, \
            f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))
        
    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        else:
            attn_mask = attn_mask + key_padding_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    ########################################################
    # So far the above code was identical to
    # torch.nn.functional._multi_head_attention_forward
    # The following sections are for calculating the 
    # attention and out projection, where the differences 
    # from the original code start.
    ########################################################
    attn_output, attn_output_weights = kernel_based_attention(
        query=q, key=k, value=v,
        kernel_function=kernel_func,
        attn_mask=attn_mask,
        is_causal=False,    # False since the causal mask is handle already, otherwise this would create a new mask
        dropout_p=dropout_p,
        need_weights=need_weights,
        batch_first=True)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

    if need_weights:
        # optionally average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        if average_attn_weights:
            attn_output_weights = attn_output_weights.mean(dim=1)  
    if not is_batched:
        # squeeze the output if input was unbatched
        attn_output = attn_output.squeeze(1)
    return attn_output, attn_output_weights