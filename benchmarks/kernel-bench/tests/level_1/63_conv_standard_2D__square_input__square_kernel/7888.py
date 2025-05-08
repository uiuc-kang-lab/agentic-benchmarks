
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build and load the custom CUDA kernel extension from "kernel.cu"
def build_kernel():
    cwd = os.path.dirname(os.path.abspath(__file__))
    cuda_file = os.path.join(cwd, "kernel.cu")
    module = load(
        name="custom_conv2d",
        sources=[cuda_file],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

# Test 1: Using groups != 1 should trigger a check error.
def test_groups_not_supported():
    cuda_module = build_kernel()
    # Create dummy input tensor of shape (N, Cin, H, W)
    N, Cin, H, W = 1, 4, 32, 32
    x = torch.randn(N, Cin, H, W, device="cuda", dtype=torch.float32)
    # Create weight with shape (Cout, Cin/groups, K, K). Here groups=2.
    groups = 2
    Cout = 4
    # For groups != 1, the weight shape would be (Cout, Cin//groups, K, K)
    K = 3
    if Cin % groups != 0:
        pytest.skip("Cin not divisible by groups, adjust parameters accordingly.")
    weight = torch.randn(Cout, Cin // groups, K, K, device="cuda", dtype=torch.float32)
    bias = torch.randn(Cout, device="cuda", dtype=torch.float32)

    stride, padding, dilation = 1, 0, 1

    with pytest.raises(RuntimeError, match="groups != 1 not supported"):
        # The forward function should check groups and raise an error.
        cuda_module.forward(x, weight, bias, stride, padding, dilation, groups)

# Test 2: Using a tensor with a wrong floating type (e.g., float64) should cause an issue.
def test_dtype_mismatch():
    cuda_module = build_kernel()
    N, Cin, H, W = 1, 3, 32, 32
    # Create input tensor with double precision.
    x = torch.randn(N, Cin, H, W, device="cuda", dtype=torch.float64)
    K = 3
    Cout = 8
    weight = torch.randn(Cout, Cin, K, K, device="cuda", dtype=torch.float64)
    bias = torch.randn(Cout, device="cuda", dtype=torch.float64)

    stride, padding, dilation, groups = 1, 0, 1, 1

    # Although CHECK_INPUT does not enforce dtype,
    # the kernel uses data_ptr<float>(), so a type mismatch is expected.
    with pytest.raises(RuntimeError):
        cuda_module.forward(x, weight, bias, stride, padding, dilation, groups)
    # Note: The error might be thrown during kernel execution due to memory misinterpretation.

# Test 3: Using non-contiguous input tensors should trigger the CHECK_INPUT failure.
def test_non_contiguous():
    cuda_module = build_kernel()
    N, Cin, H, W = 1, 3, 32, 32
    # Create a contiguous tensor and then make it non-contiguous
    x = torch.randn(N, Cin, H, W, device="cuda", dtype=torch.float32)
    x_non_contig = x.transpose(1, 2)  # This will usually be non-contiguous.
    K = 3
    Cout = 8
    weight = torch.randn(Cout, Cin, K, K, device="cuda", dtype=torch.float32)
    bias = torch.randn(Cout, device="cuda", dtype=torch.float32)

    stride, padding, dilation, groups = 1, 0, 1, 1

    with pytest.raises(RuntimeError, match="must be contiguous"):
        cuda_module.forward(x_non_contig, weight, bias, stride, padding, dilation, groups)

# Test 4: (Optional) Using the kernel launch synchronously every time can hurt performance.
# While this is not a correctness issue, we check that cudaDeviceSynchronize is present.
# Since there is no direct way to test performance in a unit test, we check that the output is computed correctly.
def test_output_correctness_float32():
    cuda_module = build_kernel()
    N, Cin, H, W = 1, 3, 16, 16
    x = torch.randn(N, Cin, H, W, device="cuda", dtype=torch.float32)
    K = 3
    Cout = 3
    # Define weight and bias such that the convolution is equivalent to a simple sum.
    weight = torch.ones(Cout, Cin, K, K, device="cuda", dtype=torch.float32)
    bias = torch.zeros(Cout, device="cuda", dtype=torch.float32)
    stride, padding, dilation, groups = 1, 1, 1, 1

    # Compute output via the custom kernel.
    output = cuda_module.forward(x, weight, bias, stride, padding, dilation, groups)
    # Compute a reference result using PyTorch's built-in convolution.
    conv = torch.nn.Conv2d(Cin, Cout, kernel_size=K, stride=stride, padding=padding, dilation=dilation, bias=True)
    conv.weight.data.fill_(1.0)
    conv.bias.data.fill_(0.0)
    # Ensure conv is on cuda and in float32.
    conv = conv.to("cuda", torch.float32)
    ref_output = conv(x)
    
    # Allow some tolerance because of potential differences in arithmetic order.
    assert torch.allclose(output, ref_output, atol=1e-5), "Custom kernel output does not match reference convolution."

if __name__ == "__main__":
    pytest.main([__file__])
