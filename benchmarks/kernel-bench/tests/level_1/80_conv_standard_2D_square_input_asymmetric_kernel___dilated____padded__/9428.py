
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# Helper function to compile and load the CUDA kernel extension.
def build_kernel():
    module = load(
        name="custom_conv2d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

# Issue 1: No error-checking after kernel launch.
# Test: Provide a noncontiguous input tensor (the forward function checks for contiguity)
def test_noncontiguous_input():
    mod = build_kernel()
    # Create a contiguous tensor and then create a noncontiguous view by transpose operation.
    x = torch.randn(16, 3, 256, 256, device="cuda")
    weight = torch.randn(64, 3, 3, 5, device="cuda")
    bias = torch.randn(64, device="cuda")
    x_noncontig = x.transpose(2, 3)
    # When passing a noncontiguous tensor, the TORCH_CHECK should trigger an error.
    with pytest.raises(RuntimeError, match="contiguous"):
        mod.forward(x_noncontig, weight, bias, 1, (1, 2), (2, 1))

# Issue 2: Batch dimension handling (the kernel loops over batches serially).
# Test: Compare the kernel output with PyTorch’s native conv2d on a multi‐batch tensor.
def test_batch_dimension():
    mod = build_kernel()
    batch_size = 32  # larger batch size to stress the serialized batch loop in the kernel
    in_channels = 3
    out_channels = 64
    height = width = 128
    kernel_size = (3, 5)
    stride = 1
    padding = (1, 2)
    dilation = (2, 1)

    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1], device="cuda", dtype=torch.float32)
    bias_tensor = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    # Compute reference using PyTorch's conv2d.
    ref = F.conv2d(x, weight, bias_tensor, stride=stride, padding=padding, dilation=dilation)
    
    # Compute kernel output.
    out = mod.forward(x, weight, bias_tensor, stride, padding, dilation)
    
    # Force synchronization to ensure any kernel launch errors are reported.
    torch.cuda.synchronize()

    # Compare outputs; even if correct, the batching strategy may affect performance.
    assert torch.allclose(out, ref, atol=1e-4), "Kernel output differs from reference for multi-batch input!"

# Issue 3: Over-aggressive unrolling assumptions in loops with runtime-dependent bounds.
# Test: Use a larger, non-standard kernel size (e.g. 7x7) with dilation/padding settings that
# differ from the hard-coded assumptions and compare with the reference conv2d.
def test_nonstandard_kernel_size():
    mod = build_kernel()
    batch_size = 8
    in_channels = 3
    out_channels = 32
    height = width = 64
    kernel_size = (7, 7)  # non-standard kernel size (greater than 3 or 5)
    stride = 1
    # Set padding and dilation such that the output dimensions are valid.
    padding = (3, 3)
    dilation = (2, 2)

    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1], device="cuda", dtype=torch.float32)
    bias_tensor = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    ref = F.conv2d(x, weight, bias_tensor, stride=stride, padding=padding, dilation=dilation)
    out = mod.forward(x, weight, bias_tensor, stride, padding, dilation)
    torch.cuda.synchronize()
    assert torch.allclose(out, ref, atol=1e-4), "Kernel output differs from reference for nonstandard kernel configuration!"
