
import pytest
import torch
from torch import nn
from torch.utils.cpp_extension import load
import math

# Utility function to build and load the CUDA extension from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="conv_transpose1d_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# A simple reference model using PyTorch's own ConvTranspose1d for comparison.
class RefModel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(RefModel, self).__init__()
        self.conv1d_transpose = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias
        )
    def forward(self, x):
        return self.conv1d_transpose(x)

# Test 1: Trigger issue with unsupported data type (e.g. double precision)
def test_non_float32_dtype():
    # Build kernel module
    module = build_kernel()

    # Create input tensors in double precision.
    batch_size = 2
    in_channels = 3
    out_channels = 4
    length = 16
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    bias = False

    x = torch.randn(batch_size, in_channels, length, device="cuda", dtype=torch.double)
    # Manually create weight tensor with shape [C_in, C_out, kernel_size]
    weight = torch.randn(in_channels, out_channels, kernel_size, device="cuda", dtype=torch.double)
    # For this test, we pass None as bias.

    # The kernel always casts pointers to float*.
    # We expect that running with double will produce incorrect results.
    y_kernel = module.forward(x, weight, None, stride, padding, dilation)
    
    # Compare against a reference implementation built with float version
    ref_model = RefModel(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)
    # Convert the double tensors to float to simulate what the kernel expects.
    y_ref = ref_model(x.float())
    
    # Since the kernel is not type-generic, the output should diverge from the reference.
    # We check that the result is not close.
    assert not torch.allclose(y_kernel, y_ref, atol=1e-4), \
        "Kernel should not support double precision but produced results similar to reference."

# Test 2: Trigger issue with grid configuration when output size is very large.
def test_excessive_grid_dimension():
    module = build_kernel()

    # We create tensors that force the output to have an extremely large number of elements,
    # which in turn would require a grid dimension that exceeds CUDA's limits.
    #
    # We set up a transposed convolution where L_out is huge.
    # L_out = (L_in - 1) * stride - 2 * padding + dilation * (K_w - 1) + 1.
    # For simplicity, choose L_in = 1, so L_out = dilation*(K_w-1)+1.
    # We choose a huge kernel size K_w and a moderate out_channels so that N * C_out * L_out exceeds
    # what can be mapped onto a one-dimensional grid.
    batch_size = 1
    in_channels = 1
    # Choose out_channels large enough to push total output elements high.
    out_channels = 1000
    L_in = 1
    # Choose kernel_size huge so that L_out gets huge.
    kernel_size = 500000  
    stride = 1
    padding = 0
    dilation = 1
    bias = False

    # Compute L_out based on formula:
    # L_out = (1 - 1) * stride - 2*padding + dilation*(kernel_size - 1) + 1 = kernel_size
    L_out = kernel_size

    try:
        # Create input x of shape (N, C_in, L_in)
        x = torch.randn(batch_size, in_channels, L_in, device="cuda", dtype=torch.float32)
        # Create weight tensor of shape (C_in, C_out, kernel_size)
        weight = torch.randn(in_channels, out_channels, kernel_size, device="cuda", dtype=torch.float32)
        
        # We expect that the kernel launch will fail due to grid dimension issues
        y = module.forward(x, weight, None, stride, padding, dilation)
        # Force synchronization to detect kernel launch errors.
        torch.cuda.synchronize()
    except RuntimeError as e:
        error_message = str(e)
        # Look for an error message that indicates failure due to grid launching or memory
        assert "CUDA kernel failed" in error_message or "invalid configuration argument" in error_message, \
            f"Unexpected error message: {error_message}"
    else:
        pytest.skip("Test skipped because the huge grid dimension was not triggered on this device.")

# Test 3: Ensure that the kernel checks for CUDA device inputs.
def test_non_cuda_inputs():
    module = build_kernel()

    batch_size = 2
    in_channels = 3
    out_channels = 4
    length = 16
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    bias = False

    # Create a CPU tensor instead of CUDA.
    x = torch.randn(batch_size, in_channels, length, device="cpu", dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels, kernel_size, device="cuda", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="Input tensor must be on CUDA device"):
        _ = module.forward(x, weight, None, stride, padding, dilation)
