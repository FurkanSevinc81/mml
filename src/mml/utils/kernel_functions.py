import torch
from torch import Tensor
from numpy import sqrt
from .ops import _correct_dim


def linear_kernel(x:Tensor, y:Tensor)->Tensor:
    """
        Computes the linear kernel as: 
            k(x, y) = x^T . y
        Args:
            x:
            y:
    """
    y = _correct_dim(y)
    return torch.matmul(x, y)

def rbf_kernel(x:Tensor, y:Tensor, gamma=None,
               stable:bool=False)->Tensor:
    """
        Computes the radial basis function (RBF) kernel as: 
            k(x, y) = exp(-gamma * ||x-y||^2)
        Args:
            x:
            y:
            gamma:
    """
    if gamma is None:
        gamma = 1.0 / x.shape[-1]

    if x.dtype == torch.float16 or y.dtype == torch.float16:
        cdist = torch.cdist(x.to(torch.float32),
                           y.to(torch.float32))
        cdist = cdist.to(dtype=torch.float16) 
    else:
        cdist = torch.cdist(x, y)

    if stable:
        return -gamma * torch.pow(cdist, 2)
    return torch.exp(-gamma * torch.pow(cdist, 2))

def polynomial_kernel(x:Tensor, y:Tensor, gamma=None, 
                      coeff=1, degree=3, stable:bool=False
                      )->Tensor:
    """
        Computes the polynomial kernel as: 
            k(x, y) = (gamma * x^T.y + c0)^d
        Args:
            x:
            y:
            gamma:
            bias:
            degree:
    """
    if gamma is None:
        gamma = 1.0 / x.shape[-1]
    y = _correct_dim(y)
    if stable: # TODO
        return None
    return torch.pow((gamma * torch.matmul(x, y) + coeff), degree)

def sigmoid_kernel(x:Tensor, y:Tensor, gamma=None,
                   coeff=1)->Tensor:
    """
        Computes the sigmoid kernel (hyperbolic tangent) as: 
            k(x, y) = tanh(gamma * x^T.y + c0)
        Args:
            x:
            y:
            gamma:
            bias:
    """
    if gamma is None:
        gamma = 1.0 / x.shape[-1]
    y = _correct_dim(y)
    return torch.tanh(gamma * torch.matmul(x, y) + coeff)

def laplacian_kernel(x:Tensor, y:Tensor, 
                     gamma=None, stable:bool=False
                     )->Tensor:
    """
        Computes the laplacian kernel as: 
            k(x, y) = exp(-gamma * ||x-y||_1)
        Args:
            x:
            y:
            gamma:
    """
    if gamma is None:
        gamma = 1.0 / x.shape[-1]

    if x.dtype == torch.float16 or y.dtype == torch.float16:
        cdist = torch.cdist(x.to(torch.float32),
                           y.to(torch.float32),
                           p=1)
        cdist = cdist.to(dtype=torch.float16) 
    else:
        cdist = torch.cdist(x, y, p=1)
    if stable:
        return -gamma * cdist
    return torch.exp(-gamma * cdist)

def exponential_kernel(x:Tensor, y:Tensor, 
                       gamma=None, stable:bool=False
                       )->Tensor:
    """
        Computes a scaled exponential kernel as:
            k(x, y) = exp(gamma * <x, y>)
    """
    if gamma is None:
        gamma = 1.0 /sqrt(x.shape[-1])
    y = _correct_dim(y)
    if stable:
        return gamma * torch.matmul(x, y)
    return torch.exp(gamma * torch.matmul(x, y))
    
class LinearKernel:
    def __init__(self, device=None, dtype=None):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        self.has_stable = False

    def __call__(self, x:Tensor, y:Tensor)->Tensor:
        x = x.to(**self.factory_kwargs)
        y = y.to(**self.factory_kwargs)
        return linear_kernel(x, y)
    
class PolynomialKernel:
    def __init__(self, degree=3, gamma=None, coeff=1, device=None, dtype=None):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        self.degree = degree
        self.gamma = gamma
        self.coeff = coeff
        self.has_stable = False

    def __call__(self, x:Tensor, y:Tensor)->Tensor:
        x = x.to(**self.factory_kwargs)
        y = y.to(**self.factory_kwargs)

        return polynomial_kernel(x, y, self.gamma, self.coeff, self.degree)

class RBFKernel:
    def __init__(self, gamma=None, device=None, dtype=None):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        self.gamma = gamma
        self.has_stable = True

    def __call__(self, x:Tensor, y:Tensor, stable:bool=False)->Tensor:
        x = x.to(**self.factory_kwargs)
        y = y.to(**self.factory_kwargs) 
        return rbf_kernel(x, y, self.gamma, stable)

class SigmoidKernel:
    def __init__(self, gamma=None, coeff=1, device=None, dtype=None):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        self.gamma = gamma
        self.coeff = coeff
        self.has_stable = False

    def __call__(self, x:Tensor, y:Tensor)->Tensor:
        x = x.to(**self.factory_kwargs)
        y = y.to(**self.factory_kwargs)
        return sigmoid_kernel(x, y, self.gamma, self.coeff)
    
class LaplacianKernel:
    def __init__(self, gamma=None, device=None, dtype=None):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        self.gamma = gamma
        self.has_stable = True

    def __call__(self, x:Tensor, y:Tensor, stable:bool=False)->Tensor:
        x = x.to(**self.factory_kwargs)
        y = y.to(**self.factory_kwargs) 
        return laplacian_kernel(x, y, self.gamma, stable)
    
class ExponentialKernel:
    def __init__(self, gamma=None, device=None, dtype=None):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        self.gamma = gamma
        self.has_stable = True

    def __call__(self, x:Tensor, y:Tensor, stable:bool=False)->Tensor:
        x = x.to(**self.factory_kwargs)
        y = y.to(**self.factory_kwargs) 
        return exponential_kernel(x, y, self.gamma, stable)

"""
def chi_squared_kernel(x, y, gamma):
    
        Computes the chi-squared kernel as: 
            k(x, y) = exp(-gamma sum( (x_i - y_i)^2 / (x_i + y_i) ))
        Args:
            x:
            y:
            gamma:
   
    #return torch.exp(-gamma, torch.sum((x-y)**2 / (x+y)))
    ...
"""