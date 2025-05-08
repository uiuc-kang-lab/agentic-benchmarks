
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper function to build the CUDA extension module
def build_kernel():
    cuda_module = load(
        name="cuda_conv2d_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

@pytest.fixture(scope="module")
def cuda_module():
    return build_kernel()

# Helper function to create input, weight, and bias tensors for convolution.
def create_conv_tensors(dtype=torch.float32, device="cuda", 
                          bias_shape=None, C_in=3, C_out=64,
                          batch_size=2, H_in=32, W_in=32, 
                          kernel_size=(3,3)):
    # Create input, weight, and optionally bias
    x = torch.randn(batch_size, C_in, H_in, W_in, dtype=dtype, device=device)
    # Weight shape: (C_out, C_in/groups, kernel_height, kernel_width)
    weight = torch.randn(C_out, C_in, kernel_size[0], kernel_size[1], dtype=dtype, device=device)
    if bias_shape is None:
        bias = torch.randn(C_out, dtype=dtype, device=device)
    else:
        bias = torch.randn(bias_shape, dtype=dtype, device=device)
    # Other parameters: stride, padding, dilation, groups
    stride = [1, 1]
    padding = [0, 0]
    dilation = [1, 1]
    groups = 1
    return x, weight, bias, stride, padding, dilation, groups

# Issue 1: Non-float32 data type handling.
def test_non_float32_dtype(cuda_module):
    # Create tensors of type double. The kernel is written for float,
    # so using double is expected to malfunction (undefined behavior).
    x, weight, bias, stride, padding, dilation, groups = create_conv_tensors(
        dtype=torch.float64, device="cuda")
    with pytest.raises(RuntimeError):
        # Expect an error or misbehavior when launching the kernel.
        out = cuda_module.forward(x, weight, bias, stride, padding, dilation, groups)
        torch.cuda.synchronize()

# Issue 2: Non-CUDA tensors.
def test_non_cuda_input(cuda_module):
    # Create tensors on the CPU.
    x, weight, bias, stride, padding, dilation, groups = create_conv_tensors(
        dtype=torch.float32, device="cpu")
    with pytest.raises(RuntimeError) as excinfo:
        out = cuda_module.forward(x, weight, bias, stride, padding, dilation, groups)
    assert "Input tensor must be on CUDA" in str(excinfo.value)

# Issue 3: Incorrect bias tensor shape.
def test_bias_incorrect_shape(cuda_module):
    # Create tensors where bias has the wrong shape (e.g., one less element than C_out)
    C_out = 64
    x, weight, _, stride, padding, dilation, groups = create_conv_tensors(
        dtype=torch.float32, device="cuda", C_out=C_out)
    wrong_bias_shape = C_out - 1  # intentionally incorrect bias shape
    wrong_bias = torch.randn(wrong_bias_shape, dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError):
        out = cuda_module.forward(x, weight, wrong_bias, stride, padding, dilation, groups)
        torch.cuda.synchronize()
