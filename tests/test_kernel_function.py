import unittest
import torch
import numpy as np
import sklearn.metrics.pairwise as prw
import src.utils.kernel_functions as ops

class TestKernelFunctions(unittest.TestCase):
    def setUp(self):
        self.X = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.Y = torch.tensor([[2, 3, 4], [5, 6, 7]])

        self.X_single = torch.rand(512, 1)
        self.Y_single = torch.rand(512, 1)

        self.X_batched = torch.rand(100, 28, 45)
        self.Y_batched = torch.rand(100, 28, 45)

    ##### Testing linear kernel against scikit implementation
    def test_linear_1(self):
        target = prw.linear_kernel(self.X, self.Y)
        pred = ops.linear_kernel(self.X, self.Y)
        self.assertTrue(torch.allclose(pred.type(torch.DoubleTensor),
                                       torch.tensor(target, dtype=torch.double),
                                       atol=1e-5))

    def test_linear_2(self):
        target = prw.linear_kernel(self.X_single, self.Y_single)
        pred = ops.linear_kernel(self.X_single, self.Y_single)
        self.assertTrue(torch.allclose(pred.type(torch.DoubleTensor),
                                       torch.tensor(target, dtype=torch.double),
                                       atol=1e-5))

    def test_linear_3(self):
        target = []
        for X, Y in zip(self.X_batched, self.Y_batched): 
            target.append(prw.linear_kernel(X, Y))
        target = np.array(target)
        pred = ops.linear_kernel(self.X_batched, self.Y_batched)
        self.assertTrue(torch.allclose(pred.type(torch.DoubleTensor),
                                       torch.tensor(target, dtype=torch.double),
                                       atol=1e-5))

    ##### Testing polynomial kernel against scikit implementation
    def test_poly_1(self):
        gamma, bias, degree = 1.5, .5, 3
        target = prw.polynomial_kernel(self.X, self.Y, degree, gamma, bias)
        pred = ops.polynomial_kernel(self.X, self.Y, gamma, bias, degree)
        self.assertTrue(torch.allclose(pred.type(torch.DoubleTensor),
                                       torch.tensor(target, dtype=torch.double),
                                       atol=1e-5))

    def test_poly_2(self):
        gamma, bias, degree = 1.5, .5, 3
        target = prw.polynomial_kernel(self.X, self.Y, degree, gamma, bias)
        pred = ops.polynomial_kernel(self.X, self.Y, gamma, bias, degree)
        self.assertTrue(torch.allclose(pred.type(torch.DoubleTensor),
                                       torch.tensor(target, dtype=torch.double), 
                                       atol=1e-5))

    def test_poly_3(self):
        gamma, bias, degree = 1.5, .5, 3
        target = []
        for X, Y in zip(self.X_batched, self.Y_batched): 
            target.append(prw.polynomial_kernel(self.X, self.Y, degree, gamma, bias))
        target = np.array(target)
        pred = ops.polynomial_kernel(self.X, self.Y, gamma, bias, degree)
        self.assertTrue(torch.allclose(pred.type(torch.DoubleTensor),
                                       torch.tensor(target, dtype=torch.double), 
                                       atol=1e-5))
        
    ##### Testing RBF kernel against scikit implementation
    def test_rbf_1(self):
        gamma = 1.5
        target = prw.rbf_kernel(self.X, self.Y, gamma)
        pred = ops.rbf_kernel(self.X, self.Y, gamma)
        self.assertTrue(torch.allclose(pred.type(torch.DoubleTensor),
                                       torch.tensor(target, dtype=torch.double),
                                       atol=1e-5))

    def test_rbf_2(self):
        gamma = 1.5
        target = prw.rbf_kernel(self.X, self.Y, gamma)
        pred = ops.rbf_kernel(self.X, self.Y, gamma)
        self.assertTrue(torch.allclose(pred.type(torch.DoubleTensor),
                                       torch.tensor(target, dtype=torch.double), 
                                       atol=1e-5))

    def test_rbf_3(self):
        gamma = 1.5
        target = []
        for X, Y in zip(self.X_batched, self.Y_batched): 
            target.append(prw.rbf_kernel(self.X, self.Y, gamma))
        target = np.array(target)
        pred = ops.rbf_kernel(self.X, self.Y, gamma)
        self.assertTrue(torch.allclose(pred.type(torch.DoubleTensor),
                                       torch.tensor(target, dtype=torch.double), 
                                       atol=1e-5))
        
    ##### Testing sigmoid kernel against scikit implementation
    def test_sigmoid_1(self):
        gamma, bias = 1.5, .5
        target = prw.sigmoid_kernel(self.X, self.Y, gamma, bias)
        pred = ops.sigmoid_kernel(self.X, self.Y, gamma, bias)
        self.assertTrue(torch.allclose(pred.type(torch.DoubleTensor),
                                       torch.tensor(target, dtype=torch.double),
                                       atol=1e-5))

    def test_sigmoid_2(self):
        gamma, bias = 1.5, .5
        target = prw.sigmoid_kernel(self.X, self.Y, gamma, bias)
        pred = ops.sigmoid_kernel(self.X, self.Y, gamma, bias)
        self.assertTrue(torch.allclose(pred.type(torch.DoubleTensor),
                                       torch.tensor(target, dtype=torch.double), 
                                       atol=1e-5))

    def test_sigmoid_3(self):
        gamma, bias = 1.5, .5
        target = []
        for X, Y in zip(self.X_batched, self.Y_batched): 
            target.append(prw.sigmoid_kernel(self.X, self.Y, gamma, bias))
        target = np.array(target)
        pred = ops.sigmoid_kernel(self.X, self.Y, gamma, bias)
        self.assertTrue(torch.allclose(pred.type(torch.DoubleTensor),
                                       torch.tensor(target, dtype=torch.double), 
                                       atol=1e-5))
    ##### Testing Laplacian kernel against scikit implementation
    def test_laplace_1(self):
        gamma = 1.5
        target = prw.laplacian_kernel(self.X, self.Y, gamma)
        pred = ops.laplacian_kernel(self.X, self.Y, gamma)
        self.assertTrue(torch.allclose(pred.type(torch.DoubleTensor),
                                       torch.tensor(target, dtype=torch.double),
                                       atol=1e-5))

    def test_laplace_2(self):
        gamma = 1.5
        target = prw.laplacian_kernel(self.X, self.Y, gamma)
        pred = ops.laplacian_kernel(self.X, self.Y, gamma)
        self.assertTrue(torch.allclose(pred.type(torch.DoubleTensor),
                                       torch.tensor(target, dtype=torch.double), 
                                       atol=1e-5))

    def test_laplace_3(self):
        gamma = 1.5
        target = []
        for X, Y in zip(self.X_batched, self.Y_batched): 
            target.append(prw.laplacian_kernel(self.X, self.Y, gamma))
        target = np.array(target)
        pred = ops.laplacian_kernel(self.X, self.Y, gamma)
        self.assertTrue(torch.allclose(pred.type(torch.DoubleTensor),
                                       torch.tensor(target, dtype=torch.double), 
                                       atol=1e-5))
        
    ##### Testing Chi-Squared kernel against scikit implementation
    def test_chi2_1(self):
        gamma = 1.5
        target = prw.chi2_kernel(self.X, self.Y, gamma)
        pred = ops.chi_squared_kernel(self.X, self.Y, gamma)
        self.assertTrue(torch.allclose(pred.type(torch.DoubleTensor),
                                       torch.tensor(target, dtype=torch.double),
                                       atol=1e-5))

    def test_chi2_2(self):
        gamma = 1.5
        target = prw.chi2_kernel(self.X, self.Y, gamma)
        pred = ops.chi_squared_kernel(self.X, self.Y, gamma)
        self.assertTrue(torch.allclose(pred.type(torch.DoubleTensor),
                                       torch.tensor(target, dtype=torch.double), 
                                       atol=1e-5))

    def test_chi2_3(self):
        gamma = 1.5
        target = []
        for X, Y in zip(self.X_batched, self.Y_batched): 
            target.append(prw.chi2_kernel(self.X, self.Y, gamma))
        target = np.array(target)
        pred = ops.chi_squared_kernel(self.X, self.Y, gamma)
        self.assertTrue(torch.allclose(pred.type(torch.DoubleTensor),
                                       torch.tensor(target, dtype=torch.double), 
                                       atol=1e-5))

if __name__ == '__main__':
    unittest.main()        
