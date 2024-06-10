import unittest
import torch
import numpy as np
import src.utils.kernel_functions as kops
import src.utils.ops as ops
import torch.nn.functional as F


class TestKernelFunctions(unittest.TestCase):
    def setUp(self):
        self.exp_kernel = kops.ExponentialKernel() # softmax kernel
        self.rbf_kernel = kops.RBFKernel()
        self.lin_kernel = kops.LinearKernel()
        self.poly_kernel = kops.PolynomialKernel()
        self.sig_kernel = kops.SigmoidKernel()
        self.lap_kernel = kops.LaplacianKernel()
        self.batch, self.sequence, self.embedding = 100, 20, 16
        self.num_heads = 8

        self.in_proj_weight = torch.randn(3 * self.embedding, self.embedding)
        self.in_proj_bias = torch.randn(3 * self.embedding)
        self.out_proj_weight = torch.randn(self.embedding, self.embedding)
        self.out_proj_bias = torch.randn(self.embedding)

        self.query = torch.rand((self.sequence, self.batch, self.embedding))
        self.key = torch.rand((self.sequence, self.batch, self.embedding))
        self.value = torch.rand((self.sequence, self.batch, self.embedding))

        self.x = torch.rand((self.batch, self.sequence, self.embedding))
        self.y = torch.rand((self.batch, self.sequence, self.embedding))
        self.z = torch.rand((self.batch, self.sequence, self.embedding))

        self.x_mh = self.x.unsqueeze(1).expand(self.batch, 
                                               self.num_heads, 
                                               self.sequence, 
                                               self.embedding)
        self.y_mh = self.y.unsqueeze(1).expand(self.batch, 
                                               self.num_heads, 
                                               self.sequence, 
                                               self.embedding)
        self.z_mh = self.z.unsqueeze(1).expand(self.batch, 
                                               self.num_heads, 
                                               self.sequence, 
                                               self.embedding)
        self.expected_attn_weights_shape = (self.batch, self.sequence, self.sequence)
        self.expected_shape = (self.sequence, self.batch, self.embedding)

    def test_attention_no_mask(self):
        expected = F.scaled_dot_product_attention(self.x, self.y, self.z)
        output, _ = ops.kernel_based_attention(self.x, self.y, self.z,
                                               self.exp_kernel, batch_first=True)
        torch.testing.assert_close(output, expected)

    def test_attention_no_mask_b(self):
        expected = F.scaled_dot_product_attention(self.x, self.y, self.z)
        output, _ = ops.kernel_based_attention(self.x, self.y, self.z,
                                               self.exp_kernel, batch_first=True)
        torch.testing.assert_close(output, expected)

    def test_attention_causal_mask(self):
        expected = F.scaled_dot_product_attention(self.x, self.y, self.z, 
                                                    is_causal=True)
        output, _ = ops.kernel_based_attention(self.x, self.y, self.z,
                                               self.exp_kernel, 
                                               is_causal=True, 
                                               batch_first=True)
        torch.testing.assert_close(output, expected)
    
    def test_attention_masked(self):
        ...

    def test_attention_mh_no_mask(self):
        expected, _ = ops.kernel_based_attention(self.x, self.y, self.z,
                                                 self.exp_kernel, batch_first=True)
        expected = expected.unsqueeze(1).expand(self.batch,
                                                self.num_heads,
                                                self.sequence, 
                                                self.embedding)
        output, _ = ops.kernel_based_attention(self.x_mh, self.y_mh, self.z_mh,
                                               self.exp_kernel, batch_first=True)
        torch.testing.assert_close(output, expected,)

    def test_attention_mh_causal_mask(self):
        expected, _ = ops.kernel_based_attention(self.x, self.y, self.z,
                                                 self.exp_kernel,
                                                 is_causal=True,
                                                 batch_first=True)
        expected = expected.unsqueeze(1).expand(self.batch,
                                                self.num_heads,
                                                self.sequence, 
                                                self.embedding)
        output, _ = ops.kernel_based_attention(self.x_mh, self.y_mh, self.z_mh,
                                               self.exp_kernel,
                                               is_causal=True, 
                                               batch_first=True)
        torch.testing.assert_close(output, expected)
    
    def test_input_shapes(self):
        """ Testing shapes
            - query: `(N, L, E)` or `(L, N, E)` where L is the target sequence length, N is the batch size, E is
            the embedding dimension.
            - key: `(N, S, E)` or `(S, N, E)`, where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - value: `(N, S, E)` or `(S, N, E)` where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
        """
        batch_second_query = self.x.transpose(1, 0)
        batch_second_key = self.y.transpose(1, 0)
        batch_second_value = self.z.transpose(1, 0)

        expected, _  = ops.kernel_based_attention(self.x, 
                                              self.y, 
                                              self.z,
                                              self.exp_kernel, 
                                              batch_first=True)
        output, _  = ops.kernel_based_attention(batch_second_query,
                                                batch_second_key,
                                                batch_second_value,
                                                self.exp_kernel,
                                                batch_first=False)
        torch.testing.assert_close(output, expected)

    def test_KMHSA_basic(self):
        """
        Test basic functionality with all required parameters.
        Since the exponential kernel function is used, the output of the 
        kernel-based MHSA has to be identical to the regular MHSA. Here
        the output of `ops.kernel_multi_head_attention_forward` is 
        compared against the expexted output generated by 
        `torch.nn.functional.multi_head_attention_forward` with the
        same parameters for both. 
        """
        output, attn_weights = ops.kernel_multi_head_attention_forward(
            kernel_func=self.exp_kernel,
            query=self.query,
            key=self.key,
            value=self.value,
            embed_dim_to_check=self.embedding,
            num_heads=self.num_heads,
            in_proj_weight=self.in_proj_weight,
            in_proj_bias=self.in_proj_bias,
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.0,
            out_proj_weight=self.out_proj_weight,
            out_proj_bias=self.out_proj_bias,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            use_separate_proj_weight=False,
            q_proj_weight=None,
            k_proj_weight=None,
            v_proj_weight=None,
            static_k=None,
            static_v=None,
            average_attn_weights=True,
            is_causal=False,
        )

        expected, expected_attn_weights = F.multi_head_attention_forward(
            query=self.query,
            key=self.key,
            value=self.value,
            embed_dim_to_check=self.embedding,
            num_heads=self.num_heads,
            in_proj_weight=self.in_proj_weight,
            in_proj_bias=self.in_proj_bias,
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.0,
            out_proj_weight=self.out_proj_weight,
            out_proj_bias=self.out_proj_bias,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            use_separate_proj_weight=False,
            q_proj_weight=None,
            k_proj_weight=None,
            v_proj_weight=None,
            static_k=None,
            static_v=None,
            average_attn_weights=True,
            is_causal=False
        )

        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(attn_weights, expected_attn_weights)

    def test_KMHSA_basic_rbf(self):
        output, attn_weights = ops.kernel_multi_head_attention_forward(
            kernel_func=self.rbf_kernel,
            query=self.query,
            key=self.key,
            value=self.value,
            embed_dim_to_check=self.embedding,
            num_heads=self.num_heads,
            in_proj_weight=self.in_proj_weight,
            in_proj_bias=self.in_proj_bias,
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.0,
            out_proj_weight=self.out_proj_weight,
            out_proj_bias=self.out_proj_bias,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            use_separate_proj_weight=False,
            q_proj_weight=None,
            k_proj_weight=None,
            v_proj_weight=None,
            static_k=None,
            static_v=None,
            average_attn_weights=True,
            is_causal=False,
        )
        torch.testing.assert_close(self.expected_shape, output.shape, check_dtype=False)
        torch.testing.assert_close(self.expected_attn_weights_shape, attn_weights.shape, check_dtype=False)

    def test_KMHSA_basic_linear(self):
        output, attn_weights = ops.kernel_multi_head_attention_forward(
            kernel_func=self.lin_kernel,
            query=self.query,
            key=self.key,
            value=self.value,
            embed_dim_to_check=self.embedding,
            num_heads=self.num_heads,
            in_proj_weight=self.in_proj_weight,
            in_proj_bias=self.in_proj_bias,
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.0,
            out_proj_weight=self.out_proj_weight,
            out_proj_bias=self.out_proj_bias,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            use_separate_proj_weight=False,
            q_proj_weight=None,
            k_proj_weight=None,
            v_proj_weight=None,
            static_k=None,
            static_v=None,
            average_attn_weights=True,
            is_causal=False,
        )
        torch.testing.assert_close(self.expected_shape, output.shape, check_dtype=False)
        torch.testing.assert_close(self.expected_attn_weights_shape, attn_weights.shape, check_dtype=False)

    def test_KMHSA_basic_polynomial(self):
        output, attn_weights = ops.kernel_multi_head_attention_forward(
            kernel_func=self.poly_kernel,
            query=self.query,
            key=self.key,
            value=self.value,
            embed_dim_to_check=self.embedding,
            num_heads=self.num_heads,
            in_proj_weight=self.in_proj_weight,
            in_proj_bias=self.in_proj_bias,
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.0,
            out_proj_weight=self.out_proj_weight,
            out_proj_bias=self.out_proj_bias,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            use_separate_proj_weight=False,
            q_proj_weight=None,
            k_proj_weight=None,
            v_proj_weight=None,
            static_k=None,
            static_v=None,
            average_attn_weights=True,
            is_causal=False,
        )
        torch.testing.assert_close(self.expected_shape, output.shape, check_dtype=False)
        torch.testing.assert_close(self.expected_attn_weights_shape, attn_weights.shape, check_dtype=False)

    def test_KMHSA_basic_sigmoid(self):
        output, attn_weights = ops.kernel_multi_head_attention_forward(
            kernel_func=self.sig_kernel,
            query=self.query,
            key=self.key,
            value=self.value,
            embed_dim_to_check=self.embedding,
            num_heads=self.num_heads,
            in_proj_weight=self.in_proj_weight,
            in_proj_bias=self.in_proj_bias,
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.0,
            out_proj_weight=self.out_proj_weight,
            out_proj_bias=self.out_proj_bias,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            use_separate_proj_weight=False,
            q_proj_weight=None,
            k_proj_weight=None,
            v_proj_weight=None,
            static_k=None,
            static_v=None,
            average_attn_weights=True,
            is_causal=False,
        )
        torch.testing.assert_close(self.expected_shape, output.shape, check_dtype=False)
        torch.testing.assert_close(self.expected_attn_weights_shape, attn_weights.shape, check_dtype=False)

    def test_KMHSA_basic_laplace(self):
        output, attn_weights = ops.kernel_multi_head_attention_forward(
            kernel_func=self.lap_kernel,
            query=self.query,
            key=self.key,
            value=self.value,
            embed_dim_to_check=self.embedding,
            num_heads=self.num_heads,
            in_proj_weight=self.in_proj_weight,
            in_proj_bias=self.in_proj_bias,
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.0,
            out_proj_weight=self.out_proj_weight,
            out_proj_bias=self.out_proj_bias,
            training=False,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            use_separate_proj_weight=False,
            q_proj_weight=None,
            k_proj_weight=None,
            v_proj_weight=None,
            static_k=None,
            static_v=None,
            average_attn_weights=True,
            is_causal=False,
        )
        torch.testing.assert_close(self.expected_shape, output.shape, check_dtype=False)
        torch.testing.assert_close(self.expected_attn_weights_shape, attn_weights.shape, check_dtype=False)
        
if __name__ == '__main__':
    unittest.main()  