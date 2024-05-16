import torch

def gaussian_kernel():
    """
        x1.x2
    """
    ...

def linear_kernel(x1:torch.Tensor, x2:torch.Tensor):
    """
        Computes the linear kernel as: 
            k(x1, x2) = x1^T . x2
        Args:
            x1:
            x2:
    """
    return torch.matmul(x1, x2.transpose(-2, -1))

def polynomial_kernel(x1, x2, gamma, coeff, degree):
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
    return (gamma * torch.matmul(x1, x2.transpose(-2, -1)) + coeff)**degree

def rbf_kernel(x1:torch.Tensor, x2:torch.Tensor, gamma):
    """
        Computes the radial basis function (RBF) kernel as: 
            k(x1, x2) = exp(-gamma * ||x-y||^2)
        Args:
            x1:
            x2:
            gamma:
    """
    x1, x2 = x1.type(torch.DoubleTensor), x2.type(torch.DoubleTensor)
    return torch.exp(-gamma * torch.cdist(x1,x2)**2)

def sigmoid_kernel(x1, x2, gamma, coeff):
    """
        Computes the sigmoid kernel as: 
            k(x1, x2) = tanh(gamma * x1^T.x2 + c0)
        Args:
            x1:
            x2:
            gamma:
            bias:
    """
    return torch.tanh(gamma * torch.matmul(x1, x2.transpose(-2, -1)) + coeff)

def laplacian_kernel(x1, x2, gamma):
    """
        Computes the laplacian kernel as: 
            k(x1, x2) = exp(-gamma * ||x-y||_1)
        Args:
            x1:
            x2:
            gamma:
    """
    return torch.exp(-gamma * torch.linalg.norm(x1 - x2, ord=1))

def chi_squared_kernel(x1, x2, gamma):
    """
        Computes the chi-squared kernel as: 
            k(x1, x2) = exp(-gamma sum( (x_i - y_i)^2 / (x_i + y_i) ))
        Args:
            x1:
            x2:
            gamma:
    """
    return torch.exp(-gamma, torch.sum((x1-x2)**2 / (x1+x2)))