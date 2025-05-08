
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build/load the CUDA extension using the kernel.cu file.
# This module will expose a "forward" function as defined in PYBIND11_MODULE.
def build_kernel():
    # Make sure compilation happens with verbose output.
    cuda_module = load(
        name="conv1d_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.fixture(scope="session")
def cuda_module():
    return build_kernel()

# Issue 1: Data type limitation (only float32 supported)
def test_dtype_not_float32(cuda_module):
    # Create an input tensor with float64 and corresponding weight
    batch_size = 4
    in_channels = 3
    out_channels = 8
    length = 64
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    groups = 1

    # Create float64 tensors on CUDA.
    x = torch.randn(batch_size, in_channels, length, dtype=torch.float64, device="cuda")
    # Weight shape is (C_out, C_in/groups, kernel_size)
    w = torch.randn(out_channels, in_channels // groups, kernel_size, dtype=torch.float64, device="cuda")
    bias = torch.randn(out_channels, dtype=torch.float64, device="cuda")
    
    with pytest.raises(RuntimeError) as excinfo:
        cuda_module.forward(x, w, bias, stride, padding, dilation, groups)
    assert "float32" in str(excinfo.value), "Kernel should assert that inputs are float32."

# Issue 2: Non-contiguous tensors are not handled
def test_non_contiguous_input(cuda_module):
    batch_size = 4
    in_channels = 3
    out_channels = 8
    length = 64
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    groups = 1

    # Create a contiguous tensor and then make it non-contiguous by transposing dims.
    x = torch.randn(batch_size, in_channels, length, device="cuda", dtype=torch.float32)
    x_noncontig = x.transpose(1, 2)  # Now shape is (batch_size, length, in_channels)
    # Make weight contiguous (weight expected shape: (C_out, C_in/groups, kernel_size))
    w = torch.randn(out_channels, in_channels // groups, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)
    
    # The extension does not check for contiguity and computes indices assuming contiguous layout.
    # We expect the output to be wrong (i.e. not match PyTorch's native conv1d) because
    # the non-contiguous layout messes with the arithmetic.
    out_kernel = cuda_module.forward(x_noncontig, w, bias, stride, padding, dilation, groups)
    
    # Use PyTorch built-in conv1d for the correct result. First, make x contiguous.
    x_contig = x_noncontig.contiguous().transpose(1, 2)
    conv = torch.nn.functional.conv1d(x_contig, w, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    
    # The outputs should differ if the kernel wrongly assumes contiguity.
    with pytest.raises(AssertionError):
        # Use a tight tolerance so that even minor differences fail the test.
        torch.testing.assert_allclose(out_kernel, conv, atol=1e-6)

# Issue 3: Group divisibility assumption
def test_groups_not_divisible(cuda_module):
    # Use in_channels and out_channels that are not multiples of groups.
    # For example, use in_channels = 3 and groups = 2.
    batch_size = 4
    in_channels = 3   # deliberately non-divisible by groups=2
    out_channels = 5  # deliberately non-divisible by groups=2
    length = 64
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    groups = 2

    # Normally, nn.Conv1d would enforce that in_channels and out_channels be divisible by groups.
    # Here we bypass that by manually creating weight with shape: (out_channels, floor(in_channels/groups), kernel_size)
    # This is not mathematically consistent with the convolution operation.
    x = torch.randn(batch_size, in_channels, length, device="cuda", dtype=torch.float32)
    # Force weight to have channels = floor(in_channels/groups)
    weight_channels = in_channels // groups
    w = torch.randn(out_channels, weight_channels, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)
    
    # Because the kernel assumes perfect divisibility, the computed output is likely incorrect.
    out_kernel = cuda_module.forward(x, w, bias, stride, padding, dilation, groups)
    
    # Compute reference result using torch.nn.functional.conv1d.
    # To bypass PyTorch's internal check for divisibility, we directly manipulate the weight shape.
    # We simulate what a user might mistakenly do.
    try:
        conv_ref = torch.nn.functional.conv1d(x, w, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    except Exception:
        conv_ref = None

    # If conv1d fails (as expected) then we check if our kernel produced some output (likely garbage)
    if conv_ref is None:
        # The output shape must be defined.
        assert out_kernel is not None, "Kernel did not produce an output even though it bypassed the divisibility check."
    else:
        # If conv1d does not error (unlikely), then the outputs should differ.
        with pytest.raises(AssertionError):
            torch.testing.assert_allclose(out_kernel, conv_ref, atol=1e-6)

# Issue 4: BlockDim.x assumption (must be a multiple of 32)
# This issue is internal to the kernel launch: the kernel hardcodes blockSize = 256.
# To simulate a scenario where blockDim.x is not a multiple of 32, one would need to override
# the launch configuration. Since our extension does not provide an interface to change blockSize,
# we include a dummy test that can be activated if the extension is modified to allow non-multiple-of-32 block sizes.
@pytest.mark.skip(reason="BlockDim.x configuration is hardcoded; cannot trigger issue through the public API.")
def test_invalid_blockDim(cuda_module):
    batch_size = 4
    in_channels = 4  # choose numbers that are divisible by groups for a normal case
    out_channels = 8
    length = 64
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    groups = 1

    x = torch.randn(batch_size, in_channels, length, device="cuda", dtype=torch.float32)
    w = torch.randn(out_channels, in_channels // groups, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)
    
    # Here we would call a modified kernel.forward that allows specifying blockDim.x.
    # For demonstration, we simply call the standard forward and expect the test to be updated later.
    out_kernel = cuda_module.forward(x, w, bias, stride, padding, dilation, groups)
    # For a true test, one would compare against a correct output.
    # We just check that an output is produced.
    assert out_kernel is not None, "Kernel did not produce an output."
