
import pytest
import torch
from torch.nn.functional import conv1d
from torch.utils.cpp_extension import load

# Helper function to build/load the CUDA extension module.
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Test case 1: Non-contiguous input triggers incorrect indexing.
def test_non_contiguous_input():
    # Create contiguous input tensors first.
    batch_size = 2
    in_channels = 4
    length = 20
    out_channels = 8
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    groups = 1

    # Create input and weight tensors in float32 on CUDA.
    x = torch.randn(batch_size, in_channels, length, device="cuda", dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels // groups, kernel_size, device="cuda", dtype=torch.float32)
    bias = None

    # Make input non-contiguous by transposing last two dims and then transposing back.
    # This will ensure a non-standard stride.
    x_noncontig = x.transpose(1, 2).clone().transpose(1, 2)
    assert not x_noncontig.is_contiguous(), "x_noncontig should be non-contiguous to trigger the bug."

    # Reference using PyTorch's conv1d (which requires contiguous inputs)
    ref_out = conv1d(x, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    
    # Use our custom kernel
    kernel_mod = build_kernel()
    out = kernel_mod.forward(x_noncontig, weight, None, stride, padding, dilation, groups)
    
    # Force device synchronization and compare results.
    torch.cuda.synchronize()
    # The outputs won’t match because our kernel’s indexing assumes contiguous layout.
    with pytest.raises(AssertionError):
        assert torch.allclose(out, ref_out, atol=1e-5), "Kernel did not produce the expected results with non-contiguous input."

# Test case 2: Wrong dtype for weight (double instead of float32) should trigger an error.
def test_wrong_dtype_weight():
    batch_size = 2
    in_channels = 4
    length = 20
    out_channels = 8
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    groups = 1

    x = torch.randn(batch_size, in_channels, length, device="cuda", dtype=torch.float32)
    # Make weight double, which is not supported by the kernel.
    weight = torch.randn(out_channels, in_channels // groups, kernel_size, device="cuda", dtype=torch.float64)
    
    kernel_mod = build_kernel()
    with pytest.raises(RuntimeError):
        # Expect the kernel to fail, for example, via a type error or undefined behavior caught by a cuda check.
        kernel_mod.forward(x, weight, None, stride, padding, dilation, groups)
    torch.cuda.synchronize()

# Test case 3: Passing a CPU bias instead of a CUDA bias should trigger an error.
def test_cpu_bias():
    batch_size = 2
    in_channels = 4
    length = 20
    out_channels = 8
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    groups = 1

    x = torch.randn(batch_size, in_channels, length, device="cuda", dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels // groups, kernel_size, device="cuda", dtype=torch.float32)
    # Create a bias on CPU.
    bias = torch.randn(out_channels, device="cpu", dtype=torch.float32)
    
    kernel_mod = build_kernel()
    with pytest.raises(RuntimeError):
        kernel_mod.forward(x, weight, bias, stride, padding, dilation, groups)
    torch.cuda.synchronize()

# Test case 4: Kernel launch without proper synchronization might hide runtime errors.
# We simulate this by providing an input configuration where the computed output length is negative.
def test_invalid_output_length():
    batch_size = 2
    in_channels = 4
    length = 5  # small length
    out_channels = 8
    kernel_size = 7  # kernel size larger than input (with no padding)
    stride = 1
    padding = 0
    dilation = 1
    groups = 1

    x = torch.randn(batch_size, in_channels, length, device="cuda", dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels // groups, kernel_size, device="cuda", dtype=torch.float32)
    
    kernel_mod = build_kernel()
    with pytest.raises(RuntimeError):
        # The conv1d host function checks L_out > 0. This should trigger a TORCH_CHECK failure.
        kernel_mod.forward(x, weight, None, stride, padding, dilation, groups)
    torch.cuda.synchronize()
