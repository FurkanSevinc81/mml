import torch
import warnings


def _to_float(tensor:torch.Tensor, precision:str='float32')->torch.Tensor:
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

def _correct_dim(tensor:torch.Tensor)->torch.Tensor:
    if tensor.dim() <= 1:
        return tensor.transpose(-1, 0)
    else:
        return tensor.transpose(-2, -1)
    
def linear_kernel(x1:torch.Tensor, 
                  x2:torch.Tensor)->torch.Tensor:
    """
        Computes the linear kernel as: 
            k(x1, x2) = x1^T . x2
        Args:
            x1:
            x2:
    """
    x2 = _correct_dim(x2)
    return torch.matmul(x1, x2)

def rbf_kernel(x1:torch.Tensor, 
               x2:torch.Tensor, 
               gamma=None)->torch.Tensor:
    """
        Computes the radial basis function (RBF) kernel as: 
            k(x1, x2) = exp(-gamma * ||x-y||^2)
        Args:
            x1:
            x2:
            gamma:
    """
    if gamma is None:
        gamma = 1.0 / x1.shape[-1]
    return torch.exp(-gamma * torch.pow(torch.cdist(x1,x2), 2))

def polynomial_kernel(x1:torch.Tensor, 
                      x2:torch.Tensor, 
                      gamma=None, 
                      coeff=1, 
                      degree=3)->torch.Tensor:
    """
        Computes the polynomial kernel as: 
            k(x1, x2) = (gamma * x1^T.x2 + c0)^d
        Args:
            x1:
            x2:
            gamma:
            bias:
            degree:
    """
    if gamma is None:
        gamma = 1.0 / x1.shape[-1]
    x2 = _correct_dim(x2)
    return torch.pow((gamma * torch.matmul(x1, x2) + coeff), degree)

def sigmoid_kernel(x1:torch.Tensor,
                   x2:torch.Tensor,
                   gamma=None,
                   coeff=1)->torch.Tensor:
    """
        Computes the sigmoid kernel (hyperbolic tangent) as: 
            k(x1, x2) = tanh(gamma * x1^T.x2 + c0)
        Args:
            x1:
            x2:
            gamma:
            bias:
    """
    if gamma is None:
        gamma = 1.0 / x1.shape[-1]
    x2 = _correct_dim(x2)
    return torch.tanh(gamma * torch.matmul(x1, x2) + coeff)

def laplacian_kernel(x1:torch.Tensor, 
                     x2:torch.Tensor, 
                     gamma=None)->torch.Tensor:
    """
        Computes the laplacian kernel as: 
            k(x1, x2) = exp(-gamma * ||x-y||_1)
        Args:
            x1:
            x2:
            gamma:
    """
    if gamma is None:
        gamma = 1.0 / x1.shape[-1]
    return torch.exp(-gamma * torch.cdist(x1, x2, p=1))

def exponential_kernel(x1:torch.Tensor, 
                       x2:torch.Tensor, 
                       gamma=None)->torch.Tensor:
    """
        Computes a scaled exponential kernel as:
            k(x1, x2) = exp(gamma * <x1, x2>)
    """
    if gamma is None:
        gamma = 1.0 /x1.shape[-1]
    x2 = _correct_dim(x2)
    return torch.exp(gamma * torch.matmul(x1, x2))
    
class LinearKernel:
    def __init__(self, fpPrecision='float32'):
        self.fpPrecision = fpPrecision

    def __call__(self, x:torch.Tensor, y:torch.Tensor)->torch.Tensor:
        x = _to_float(x, self.fpPrecision)
        y = _to_float(y, self.fpPrecision)
        return linear_kernel(x, y)
    
class PolynomialKernel:
    def __init__(self, degree=3, gamma=None, coeff=1, fpPrecision='float32'):
        self.fpPrecision = fpPrecision
        self.degree = degree
        self.gamma = gamma
        self.coeff = coeff

    def __call__(self, x:torch.Tensor, y:torch.Tensor)->torch.Tensor:
        x = _to_float(x, self.fpPrecision)
        y = _to_float(y, self.fpPrecision)

        return polynomial_kernel(x, y, self.gamma, self.coeff, self.degree)

class RBFKernel:
    def __init__(self, gamma=None, fpPrecision='float32'):
        self.fpPrecision = fpPrecision
        self.gamma = gamma

    def __call__(self, x:torch.Tensor, y:torch.Tensor)->torch.Tensor:
        x = _to_float(x, self.fpPrecision)
        y = _to_float(y, self.fpPrecision) 
        return rbf_kernel(x, y, self.gamma)

class SigmoidKernel:
    def __init__(self, gamma=None, coeff=1, fpPrecision='float32'):
        self.fpPrecision = fpPrecision
        self.gamma = gamma
        self.coeff = coeff

    def __call__(self, x:torch.Tensor, y:torch.Tensor)->torch.Tensor:
        x = _to_float(x, self.fpPrecision)
        y = _to_float(y, self.fpPrecision)
        return sigmoid_kernel(x, y, self.gamma, self.coeff)
    
class LaplacianKernel:
    def __init__(self, gamma=None, fpPrecision='float32'):
        self.fpPrecision = fpPrecision
        self.gamma = gamma

    def __call__(self, x:torch.Tensor, y:torch.Tensor)->torch.Tensor:
        x = _to_float(x, self.fpPrecision)
        y = _to_float(y, self.fpPrecision) 
        return laplacian_kernel(x, y, self.gamma)
    
class ExponentialKernel:
    def __init__(self, gamma=None, fpPrecision='float32'):
        self.fpPrecision = fpPrecision
        self.gamma = gamma

    def __call__(self, x:torch.Tensor, y:torch.Tensor)->torch.Tensor:
        x = _to_float(x, self.fpPrecision)
        y = _to_float(y, self.fpPrecision) 
        return exponential_kernel(x, y, self.gamma)


def chi_squared_kernel(x1, x2, gamma):
    """
        Computes the chi-squared kernel as: 
            k(x1, x2) = exp(-gamma sum( (x_i - y_i)^2 / (x_i + y_i) ))
        Args:
            x1:
            x2:
            gamma:
   """
    #return torch.exp(-gamma, torch.sum((x1-x2)**2 / (x1+x2)))
    ...
