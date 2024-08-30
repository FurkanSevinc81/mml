import unittest
import torch
import numpy as np
import sklearn.metrics.pairwise as prw
import mml.utils.kernel_functions as kops

class TestKernelFunctions(unittest.TestCase):
    def setUp(self):
        self.X = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.Y = torch.tensor([[2, 3, 4], [5, 6, 7]])

        self.X_single = torch.rand(512, 1)
        self.Y_single = torch.rand(512, 1)

        self.X_vec = torch.rand(512, 10)
        self.Y_vec = torch.rand(512, 10)

        self.X_batched = torch.rand(100, 28, 45)
        self.Y_batched = torch.rand(100, 28, 45)
        
        self.gamma = None
        self.degree = 3
        self.coeff = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float
        self.factory_kwargs = {'device': self.device, 'dtype': self.dtype}

        self.linear_kernel_func = kops.LinearKernel(**self.factory_kwargs)
        self.rbf_kernel_func = kops.RBFKernel(gamma=self.gamma, 
                                             **self.factory_kwargs)
        self.polynomial_kernel_func = kops.PolynomialKernel(degree=self.degree,
                                                           gamma=self.gamma,
                                                           coeff=self.coeff,
                                                           **self.factory_kwargs)
        self.sigmoid_kernel_func = kops.SigmoidKernel(gamma=self.gamma,
                                                     coeff=self.coeff,
                                                     **self.factory_kwargs)
        self.laplacian_kernel_func = kops.LaplacianKernel(gamma=self.gamma,
                                                         **self.factory_kwargs)

    ##### Testing linear kernel against scikit implementation
    def test_linear_1(self):
        expected = prw.linear_kernel(self.X, self.Y)
        actual = self.linear_kernel_func(self.X, self.Y)
        torch.testing.assert_close(torch.tensor(expected, dtype=self.dtype),
                                   actual)

    def test_linear_2(self):
        expected = prw.linear_kernel(self.X_single, self.Y_single)
        actual = self.linear_kernel_func(self.X_single, self.Y_single)
        torch.testing.assert_close(torch.tensor(expected, dtype=self.dtype),
                                   actual)

    def test_linear_3(self):
        expected = []
        for X, Y in zip(self.X_batched, self.Y_batched): 
            expected.append(prw.linear_kernel(X, Y))
        expected = np.array(expected)
        actual = self.linear_kernel_func(self.X_batched, self.Y_batched)
        torch.testing.assert_close(torch.tensor(expected, dtype=self.dtype),
                                   actual)
        
    def test_linear_4(self):
        expected = prw.linear_kernel(self.X_vec, self.Y_vec)
        actual = self.linear_kernel_func(self.X_vec, self.Y_vec)
        torch.testing.assert_close(torch.tensor(expected, dtype=self.dtype),
                                   actual)

    ##### Testing polynomial kernel against scikit implementation
    def test_poly_1(self):
        expected = prw.polynomial_kernel(self.X, 
                                       self.Y, 
                                       self.degree, 
                                       self.gamma, 
                                       self.coeff)
        actual = self.polynomial_kernel_func(self.X, self.Y)
        torch.testing.assert_close(torch.tensor(expected, dtype=self.dtype),
                                   actual)

    def test_poly_2(self):
        expected = prw.polynomial_kernel(self.X_single, 
                                       self.Y_single, 
                                       self.degree, 
                                       self.gamma, 
                                       self.coeff)
        actual = self.polynomial_kernel_func(self.X_single, self.Y_single)
        torch.testing.assert_close(torch.tensor(expected, dtype=self.dtype), 
                                   actual)

    def test_poly_3(self):
        expected = []
        for X, Y in zip(self.X_batched, self.Y_batched): 
            expected.append(prw.polynomial_kernel(X, 
                                                Y, 
                                                self.degree,
                                                self.gamma, 
                                                self.coeff))
        expected = np.array(expected)
        actual = self.polynomial_kernel_func(self.X_batched, self.Y_batched)
        torch.testing.assert_close(torch.tensor(expected, dtype=self.dtype),
                                   actual)

    def test_poly_4(self):
        expected = prw.polynomial_kernel(self.X_vec, 
                                       self.Y_vec, 
                                       self.degree, 
                                       self.gamma, 
                                       self.coeff)
        actual = self.polynomial_kernel_func(self.X_vec, self.Y_vec)
        torch.testing.assert_close(torch.tensor(expected, dtype=self.dtype),
                                   actual)
        
    ##### Testing RBF kernel against scikit implementation
    def test_rbf_1(self):
        expected = prw.rbf_kernel(self.X, self.Y, self.gamma)
        actual = self.rbf_kernel_func(self.X, self.Y) 
        torch.testing.assert_close(torch.tensor(expected, dtype=self.dtype),
                                   actual)

    def test_rbf_2(self):
        expected = prw.rbf_kernel(self.X_single, 
                                self.Y_single, 
                                self.gamma)
        actual = self.rbf_kernel_func(self.X_single, self.Y_single)
        torch.testing.assert_close(torch.tensor(expected, dtype=self.dtype),
                                   actual)

    def test_rbf_3(self):
        expected = []
        for X, Y in zip(self.X_batched, self.Y_batched): 
            expected.append(prw.rbf_kernel(X, Y, self.gamma))
        expected = np.array(expected)
        actual = self.rbf_kernel_func(self.X_batched, self.Y_batched)
        torch.testing.assert_close(torch.tensor(expected, dtype=self.dtype),
                                   actual)
    
    def test_rbf_4(self):
        expected = prw.rbf_kernel(self.X_vec, 
                                self.Y_vec, 
                                self.gamma)
        actual = self.rbf_kernel_func(self.X_vec, self.Y_vec)
        torch.testing.assert_close(torch.tensor(expected, dtype=self.dtype),
                                   actual)

    ##### Testing sigmoid kernel against scikit implementation
    def test_sigmoid_1(self):
        expected = prw.sigmoid_kernel(self.X, self.Y, self.gamma, self.coeff)
        actual = self.sigmoid_kernel_func(self.X, self.Y)
        torch.testing.assert_close(torch.tensor(expected, dtype=self.dtype),
                                   actual)

    def test_sigmoid_2(self):
        expected = prw.sigmoid_kernel(self.X_single, 
                                    self.Y_single, 
                                    self.gamma, 
                                    self.coeff)
        actual = self.sigmoid_kernel_func(self.X_single, self.Y_single)
        torch.testing.assert_close(torch.tensor(expected, dtype=self.dtype),
                                   actual)

    def test_sigmoid_3(self):
        expected = []
        for X, Y in zip(self.X_batched, self.Y_batched): 
            expected.append(prw.sigmoid_kernel(X, Y, self.gamma, self.coeff))
        expected = np.array(expected)
        actual = self.sigmoid_kernel_func(self.X_batched, self.Y_batched)
        torch.testing.assert_close(torch.tensor(expected, dtype=self.dtype),
                                   actual)
        
    def test_sigmoid_4(self):
        expected = prw.sigmoid_kernel(self.X_vec, 
                                    self.Y_vec, 
                                    self.gamma, 
                                    self.coeff)
        actual = self.sigmoid_kernel_func(self.X_vec, self.Y_vec)
        torch.testing.assert_close(torch.tensor(expected, dtype=self.dtype),
                                   actual)
    ##### Testing Laplacian kernel against scikit implementation
    def test_laplace_1(self):
        expected = prw.laplacian_kernel(self.X, 
                                      self.Y, 
                                      self.gamma)
        actual = self.laplacian_kernel_func(self.X, self.Y)
        torch.testing.assert_close(torch.tensor(expected, dtype=self.dtype),
                                   actual)

    def test_laplace_2(self):
        expected = prw.laplacian_kernel(self.X_single, 
                                      self.Y_single, 
                                      self.gamma)
        actual = self.laplacian_kernel_func(self.X_single, self.Y_single)
        torch.testing.assert_close(torch.tensor(expected, dtype=self.dtype),
                                   actual)

    def test_laplace_3(self):
        expected = []
        for X, Y in zip(self.X_batched, self.Y_batched): 
            expected.append(prw.laplacian_kernel(X, Y, self.gamma))
        expected = np.array(expected)
        actual = self.laplacian_kernel_func(self.X_batched, self.Y_batched)
        torch.testing.assert_close(torch.tensor(expected, dtype=self.dtype),
                                   actual)
        
    def test_laplace_4(self):
        expected = prw.laplacian_kernel(self.X_vec, 
                                      self.Y_vec, 
                                      self.gamma)
        actual = self.laplacian_kernel_func(self.X_vec, self.Y_vec)
        torch.testing.assert_close(torch.tensor(expected, dtype=self.dtype),
                                   actual) 
"""    
    ##### Testing Chi-Squared kernel against scikit implementation
    def test_chi2_1(self):
        gamma = 1.5
        expected = prw.chi2_kernel(self.X, self.Y, gamma)
        actual = ops.chi_squared_kernel(self.X, self.Y, gamma)
        self.assertTrue(torch.allclose(actual.type(torch.DoubleTensor),
                                       torch.tensor(expected, dtype=torch.double),
                                       atol=self.atol))

    def test_chi2_2(self):
        gamma = 1.5
        expected = prw.chi2_kernel(self.X, self.Y, gamma)
        actual = ops.chi_squared_kernel(self.X, self.Y, gamma)
        self.assertTrue(torch.allclose(actual.type(torch.DoubleTensor),
                                       torch.tensor(expected, dtype=torch.double), 
                                       atol=self.atol))

    def test_chi2_3(self):
        gamma = 1.5
        expected = []
        for X, Y in zip(self.X_batched, self.Y_batched): 
            expected.append(prw.chi2_kernel(self.X, self.Y, gamma))
        expected = np.array(expected)
        actual = ops.chi_squared_kernel(self.X, self.Y, gamma)
        self.assertTrue(torch.allclose(actual.type(torch.DoubleTensor),
                                       torch.tensor(expected, dtype=torch.double), 
                                       atol=self.atol))
"""
if __name__ == '__main__':
    unittest.main()        
