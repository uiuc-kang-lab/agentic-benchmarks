
import pytest
import torch
from torch.utils.cpp_extension import load
import torch.nn.functional as F

def build_kernel():
    # Build the extension from kernel.cu
    module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

# Issue 1: Weight constant memory size limitation
def test_weight_constant_memory_limit():
    cuda_module = build_kernel()
    # Create input tensor
    N, C_in, H_in, W_in = 1, 3, 16, 16
    input_tensor = torch.randn(N, C_in, H_in, W_in, device="cuda", dtype=torch.float32)
    # Create a weight tensor that exceeds 8192 elements.
    # For example, shape (C_out, C_in/groups, K_h, K_w) with product > 8192.
    # Let's choose C_out=64, and K_h=16, K_w=16 gives: 64 * 3 * 16 * 16 = 49152 > 8192.
    C_out = 64
    K_h, K_w = 16, 16
    weight = torch.randn(C_out, C_in, K_h, K_w, device="cuda", dtype=torch.float32)
    bias = None

    with pytest.raises(RuntimeError, match="Weight tensor too large for constant memory"):
        # Expected to fail the TORCH_CHECK on weight.numel() <= 8192.
        cuda_module.forward(input_tensor, weight, bias, [1, 1], [0, 0], [1, 1], 1)

# Issue 2: Bias constant memory size limitation
def test_bias_constant_memory_limit():
    cuda_module = build_kernel()
    # Create input tensor
    N, C_in, H_in, W_in = 1, 3, 16, 16
    input_tensor = torch.randn(N, C_in, H_in, W_in, device="cuda", dtype=torch.float32)
    # Create a weight tensor that fits constant mem (e.g., shape that yields less than 8192 elements)
    C_out = 8
    K_h, K_w = 3, 3
    weight = torch.randn(C_out, C_in, K_h, K_w, device="cuda", dtype=torch.float32)
    # Create bias tensor with too many elements for constant memory.
    bias = torch.randn(C_out * 2, device="cuda", dtype=torch.float32)  # Exceeds 1024 if C_out*2 > 1024
    # To be sure, make a large bias tensor.
    bias = torch.randn(2048, device="cuda", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="Bias tensor too large for constant memory"):
        cuda_module.forward(input_tensor, weight, bias, [1, 1], [0, 0], [1, 1], 1)

# Issue 3: Lack of support for non-float32 (data type handling)
def test_non_float32_dtype():
    cuda_module = build_kernel()
    # Create input tensor with float64 instead of float32.
    N, C_in, H_in, W_in = 1, 3, 16, 16
    input_tensor = torch.randn(N, C_in, H_in, W_in, device="cuda", dtype=torch.float64)
    C_out = 8
    K_h, K_w = 3, 3
    weight = torch.randn(C_out, C_in, K_h, K_w, device="cuda", dtype=torch.float64)
    bias = None

    # Since the kernel uses data_ptr<float>(), using float64 should lead to incorrect results.
    # We'll compare against PyTorch's own convolution and expect a noticeable difference.
    output_kernel = cuda_module.forward(input_tensor.float(), weight.float(), bias, [1, 1], [0, 0], [1, 1], 1)
    output_reference = F.conv2d(input_tensor.double(), weight.double(), bias, stride=(1, 1), padding=(0, 0), dilation=(1, 1))
    
    # Convert kernel output to double for fair comparison.
    output_kernel_double = output_kernel.double()
    # They must not be allclose due to improper interpretation of double data.
    assert not torch.allclose(output_kernel_double, output_reference, atol=1e-5), "Kernel did not catch non-float32 dtype issue!"

# Issue 4: Weights persistence / constant memory being overwritten every call (lack of dynamic handling)
def test_constant_memory_overwrite():
    cuda_module = build_kernel()
    # Create two different weight sets that are within constant memory limits.
    N, C_in, H_in, W_in = 1, 3, 16, 16
    input_tensor = torch.randn(N, C_in, H_in, W_in, device="cuda", dtype=torch.float32)
    C_out = 4
    K_h, K_w = 3, 3
    # Weight set 1: all ones.
    weight1 = torch.ones(C_out, C_in, K_h, K_w, device="cuda", dtype=torch.float32)
    # Weight set 2: all twos.
    weight2 = torch.full((C_out, C_in, K_h, K_w), 2.0, device="cuda", dtype=torch.float32)
    bias = None

    # First call with weight1.
    output1 = cuda_module.forward(input_tensor, weight1, bias, [1, 1], [0, 0], [1, 1], 1)
    # Second call with weight2.
    output2 = cuda_module.forward(input_tensor, weight2, bias, [1, 1], [0, 0], [1, 1], 1)
    
    # Since the constant memory is overwritten on every call, the outputs should differ.
    # If the constant memory is not updated correctly, the second output may still reflect weight1.
    assert not torch.allclose(output1, output2), "Kernel constant memory was not updated between calls!"

