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

        self.X_vec = torch.rand(512)
        self.Y_vec = torch.rand(512)

        self.X_batched = torch.rand(100, 28, 45)
        self.Y_batched = torch.rand(100, 28, 45)

        self.atol = 1e-3
        self.gamma = None
        self.degree = 3
        self.coeff = 1

        self.linear_kernel_func = ops.LinearKernel(fpPrecision='float16')
        self.rbf_kernel_func = ops.RBFKernel(gamma=self.gamma, 
                                             fpPrecision='float16')
        self.polynomial_kernel_func = ops.PolynomialKernel(degree=self.degree,
                                                           gamma=self.gamma,
                                                           coeff=self.coeff,
                                                           fpPrecision='float16')
        self.sigmoid_kernel_func = ops.SigmoidKernel(gamma=self.gamma,
                                                     coeff=self.coeff,
                                                     fpPrecision='float16')
        self.laplacian_kernel_func = ops.LaplacianKernel(gamma=self.gamma,
                                                         fpPrecision='float16')

    ##### Testing linear kernel against scikit implementation
    def test_linear_1(self):
        target = prw.linear_kernel(self.X, self.Y)
        pred = self.linear_kernel_func(self.X, self.Y)
        self.assertTrue(torch.allclose(pred,
                                       torch.tensor(target, dtype=torch.float),
                                       atol=self.atol))

    def test_linear_2(self):
        target = prw.linear_kernel(self.X_single, self.Y_single)
        pred = self.linear_kernel_func(self.X_single, self.Y_single)
        self.assertTrue(torch.allclose(pred,
                                       torch.tensor(target, dtype=torch.float),
                                       atol=self.atol))

    def test_linear_3(self):
        target = []
        for X, Y in zip(self.X_batched, self.Y_batched): 
            target.append(prw.linear_kernel(X, Y))
        target = np.array(target)
        pred = self.linear_kernel_func(self.X_batched, self.Y_batched)
        self.assertTrue(torch.allclose(pred,
                                       torch.tensor(target, dtype=torch.float),
                                       atol=self.atol))
        
    def test_linear_4(self):
        target = prw.linear_kernel(self.X_vec, self.Y_vec)
        pred = self.linear_kernel_func(self.X_vec, self.Y_vec)
        self.assertTrue(torch.allclose(pred,
                                       torch.tensor(target, dtype=torch.float),
                                       atol=self.atol))

    ##### Testing polynomial kernel against scikit implementation
    def test_poly_1(self):
        target = prw.polynomial_kernel(self.X, 
                                       self.Y, 
                                       self.degree, 
                                       self.gamma, 
                                       self.coeff)
        pred = self.polynomial_kernel_func(self.X, self.Y)
        self.assertTrue(torch.allclose(pred,
                                       torch.tensor(target, dtype=torch.float),
                                       atol=self.atol))

    def test_poly_2(self):
        target = prw.polynomial_kernel(self.X_single, 
                                       self.Y_single, 
                                       self.degree, 
                                       self.gamma, 
                                       self.coeff)
        pred = self.polynomial_kernel_func(self.X_single, self.Y_single)
        self.assertTrue(torch.allclose(pred,
                                       torch.tensor(target, dtype=torch.float), 
                                       atol=self.atol))

    def test_poly_3(self):
        target = []
        for X, Y in zip(self.X_batched, self.Y_batched): 
            target.append(prw.polynomial_kernel(X, 
                                                Y, 
                                                self.degree,
                                                self.gamma, 
                                                self.coeff))
        target = np.array(target)
        pred = self.polynomial_kernel_func(self.X_batched, self.Y_batched)
        self.assertTrue(torch.allclose(pred,
                                       torch.tensor(target, dtype=torch.float), 
                                       atol=self.atol))

    def test_poly_4(self):
        target = prw.polynomial_kernel(self.X_vec, 
                                       self.Y_vec, 
                                       self.degree, 
                                       self.gamma, 
                                       self.coeff)
        pred = self.polynomial_kernel_func(self.X_vec, self.Y_vec)
        self.assertTrue(torch.allclose(pred,
                                       torch.tensor(target, dtype=torch.float), 
                                       atol=self.atol))
        
    ##### Testing RBF kernel against scikit implementation
    def test_rbf_1(self):
        target = prw.rbf_kernel(self.X, self.Y, self.gamma)
        pred = self.rbf_kernel_func(self.X, self.Y) 
        self.assertTrue(torch.allclose(pred,
                                       torch.tensor(target, dtype=torch.float),
                                       atol=self.atol))

    def test_rbf_2(self):
        target = prw.rbf_kernel(self.X_single, 
                                self.Y_single, 
                                self.gamma)
        pred = self.rbf_kernel_func(self.X_single, self.Y_single)
        self.assertTrue(torch.allclose(pred,
                                       torch.tensor(target, dtype=torch.float), 
                                       atol=self.atol))

    def test_rbf_3(self):
        target = []
        for X, Y in zip(self.X_batched, self.Y_batched): 
            target.append(prw.rbf_kernel(X, Y, self.gamma))
        target = np.array(target)
        pred = self.rbf_kernel_func(self.X_batched, self.Y_batched)
        self.assertTrue(torch.allclose(pred,
                                       torch.tensor(target, dtype=torch.float), 
                                       atol=self.atol))
    
    def test_rbf_4(self):
        target = prw.rbf_kernel(self.X_vec, 
                                self.Y_vec, 
                                self.gamma)
        pred = self.rbf_kernel_func(self.X_vec, self.Y_vec)
        self.assertTrue(torch.allclose(pred,
                                       torch.tensor(target, dtype=torch.float), 
                                       atol=self.atol))

    ##### Testing sigmoid kernel against scikit implementation
    def test_sigmoid_1(self):
        target = prw.sigmoid_kernel(self.X, self.Y, self.gamma, self.coeff)
        pred = self.sigmoid_kernel_func(self.X, self.Y)
        self.assertTrue(torch.allclose(pred,
                                       torch.tensor(target, dtype=torch.float),
                                       atol=self.atol))

    def test_sigmoid_2(self):
        target = prw.sigmoid_kernel(self.X_single, 
                                    self.Y_single, 
                                    self.gamma, 
                                    self.coeff)
        pred = self.sigmoid_kernel_func(self.X_single, self.Y_single)
        self.assertTrue(torch.allclose(pred,
                                       torch.tensor(target, dtype=torch.float), 
                                       atol=self.atol))

    def test_sigmoid_3(self):
        target = []
        for X, Y in zip(self.X_batched, self.Y_batched): 
            target.append(prw.sigmoid_kernel(X, Y, self.gamma, self.coeff))
        target = np.array(target)
        pred = self.sigmoid_kernel_func(self.X_batched, self.Y_batched)
        self.assertTrue(torch.allclose(pred,
                                       torch.tensor(target, dtype=torch.float), 
                                       atol=self.atol))
        
    def test_sigmoid_4(self):
        target = prw.sigmoid_kernel(self.X_vec, 
                                    self.Y_vec, 
                                    self.gamma, 
                                    self.coeff)
        pred = self.sigmoid_kernel_func(self.X_vec, self.Y_vec)
        self.assertTrue(torch.allclose(pred,
                                       torch.tensor(target, dtype=torch.float), 
                                       atol=self.atol))
    ##### Testing Laplacian kernel against scikit implementation
    def test_laplace_1(self):
        target = prw.laplacian_kernel(self.X, 
                                      self.Y, 
                                      self.gamma)
        pred = self.laplacian_kernel_func(self.X, self.Y)
        self.assertTrue(torch.allclose(pred,
                                       torch.tensor(target, dtype=torch.float),
                                       atol=self.atol))

    def test_laplace_2(self):
        target = prw.laplacian_kernel(self.X_single, 
                                      self.Y_single, 
                                      self.gamma)
        pred = self.laplacian_kernel_func(self.X_single, self.Y_single)
        self.assertTrue(torch.allclose(pred,
                                       torch.tensor(target, dtype=torch.float), 
                                       atol=self.atol))

    def test_laplace_3(self):
        target = []
        for X, Y in zip(self.X_batched, self.Y_batched): 
            target.append(prw.laplacian_kernel(X, Y, self.gamma))
        target = np.array(target)
        pred = self.laplacian_kernel_func(self.X_batched, self.Y_batched)
        self.assertTrue(torch.allclose(pred,
                                       torch.tensor(target, dtype=torch.float), 
                                       atol=self.atol))
        
    def test_laplace_4(self):
        target = prw.laplacian_kernel(self.X_vec, 
                                      self.Y_vec, 
                                      self.gamma)
        pred = self.laplacian_kernel_func(self.X_vec, self.Y_vec)
        self.assertTrue(torch.allclose(pred,
                                       torch.tensor(target, dtype=torch.float), 
                                       atol=self.atol))
"""    
    ##### Testing Chi-Squared kernel against scikit implementation
    def test_chi2_1(self):
        gamma = 1.5
        target = prw.chi2_kernel(self.X, self.Y, gamma)
        pred = ops.chi_squared_kernel(self.X, self.Y, gamma)
        self.assertTrue(torch.allclose(pred.type(torch.DoubleTensor),
                                       torch.tensor(target, dtype=torch.double),
                                       atol=self.atol))

    def test_chi2_2(self):
        gamma = 1.5
        target = prw.chi2_kernel(self.X, self.Y, gamma)
        pred = ops.chi_squared_kernel(self.X, self.Y, gamma)
        self.assertTrue(torch.allclose(pred.type(torch.DoubleTensor),
                                       torch.tensor(target, dtype=torch.double), 
                                       atol=self.atol))

    def test_chi2_3(self):
        gamma = 1.5
        target = []
        for X, Y in zip(self.X_batched, self.Y_batched): 
            target.append(prw.chi2_kernel(self.X, self.Y, gamma))
        target = np.array(target)
        pred = ops.chi_squared_kernel(self.X, self.Y, gamma)
        self.assertTrue(torch.allclose(pred.type(torch.DoubleTensor),
                                       torch.tensor(target, dtype=torch.double), 
                                       atol=self.atol))
"""
if __name__ == '__main__':
    unittest.main()        
