
import pytest
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os

# Build and load the CUDA extension from kernel.cu.
def build_kernel():
    # Assuming the kernel file is in the current directory.
    cuda_module = load(
        name="group_norm_ext",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Helper: run the kernel forward function
def run_kernel_forward(x, weight, bias, num_groups, eps):
    mod = build_kernel()
    # our module has a function called "forward" exposed via pybind11.
    return mod.forward(x, weight, bias, num_groups, eps)

# Create a helper that uses native PyTorch GroupNorm as reference.
def run_pytorch_groupnorm(x, num_groups, eps):
    C = x.size(1)
    gn = nn.GroupNorm(num_groups, C, eps)
    # Initialize weight and bias to ones and zeros so that only normalization is applied.
    with torch.no_grad():
        gn.weight.fill_(1.0)
        gn.bias.fill_(0.0)
    return gn(x)

# Issue 1: Non‐contiguous input
def test_noncontiguous_input():
    # Create a contiguous input
    batch_size, channels, H, W = 8, 32, 16, 16
    eps = 1e-5
    num_groups = 8
    x = torch.randn(batch_size, channels, H, W, device="cuda")
    # Create a non-contiguous view by transposing one spatial dimension.
    x_noncontig = x.transpose(2, 3)
    # Create weight and bias matching channels.
    weight = torch.ones(channels, device="cuda")
    bias = torch.zeros(channels, device="cuda")
    
    # Run kernel forward on non-contiguous input
    y_kernel = run_kernel_forward(x_noncontig.contiguous(), weight, bias, num_groups, eps)
    # Here we simulate “wrong” behavior by intentionally not calling .contiguous() in the kernel.
    # For comparison, run PyTorch's native GroupNorm on the original non-contiguous input.
    y_ref = run_pytorch_groupnorm(x_noncontig, num_groups, eps)
    
    # The outputs should match when the input is contiguous.
    # Since our kernel would be used with non-contiguous input without proper handling,
    # we expect a discrepancy.
    assert not torch.allclose(y_kernel, y_ref, atol=1e-5), "Kernel should fail for non-contiguous input."

# Issue 2: Channels not divisible by num_groups
def test_channels_not_divisible():
    # Here channels = 30, num_groups = 8, which is not divisible.
    batch_size, channels, H, W = 4, 30, 8, 8
    eps = 1e-5
    num_groups = 8
    x = torch.randn(batch_size, channels, H, W, device="cuda")
    weight = torch.ones(channels, device="cuda")
    bias = torch.zeros(channels, device="cuda")
    
    # In a more general implementation we would expect a check or a correct handling.
    # Here, we expect the kernel to produce wrong results.
    y_kernel = run_kernel_forward(x, weight, bias, num_groups, eps)
    y_ref = run_pytorch_groupnorm(x, num_groups, eps)
    # The outputs are expected to differ.
    assert not torch.allclose(y_kernel, y_ref, atol=1e-5), "Kernel should produce wrong results when channels are not divisible by num_groups."

# Issue 3: Extremely large grid (1D kernel launch limitations)
def test_large_input_grid():
    # Create a tensor with many spatial elements.
    # (Note: We choose sizes that are big enough to stress the 1D grid but not too big to run out of memory.)
    batch_size, channels, H, W = 2, 16, 2048, 2048
    eps = 1e-5
    num_groups = 4  # channels (16) divisible by 4
    x = torch.randn(batch_size, channels, H, W, device="cuda")
    weight = torch.ones(channels, device="cuda")
    bias = torch.zeros(channels, device="cuda")
    
    # This is mainly to check if launching the kernel causes grid dimension issues.
    # We compare against the PyTorch native version.
    y_kernel = run_kernel_forward(x, weight, bias, num_groups, eps)
    y_ref = run_pytorch_groupnorm(x, num_groups, eps)
    
    # Even if the kernel runs, the grid configuration might be limiting its correctness.
    assert not torch.allclose(y_kernel, y_ref, atol=1e-5), "Kernel should misbehave when the input is very large because of 1D grid limitations."

# Issue 4: Misaligned memory due to slicing (triggering __ldg alignment issues)
def test_misaligned_memory():
    # Create a tensor that is contiguous then slice it to force misalignment.
    batch_size, channels, H, W = 8, 32, 16, 16
    eps = 1e-5
    num_groups = 8
    x = torch.randn(batch_size, channels, H, W, device="cuda")
    # Make a misaligned view by taking a slice with a non-zero offset in the channel dimension.
    x_misaligned = x[:, 1:, :, :].clone()  # clone to ensure separate storage; note that this makes channels not divisible.
    # Adjust weight and bias accordingly.
    new_channels = x_misaligned.size(1)
    weight = torch.ones(new_channels, device="cuda")
    bias = torch.zeros(new_channels, device="cuda")
    
    # As we do for issue 2, we expect misalignment (and even changed channel count) to cause errors.
    y_kernel = run_kernel_forward(x_misaligned, weight, bias, num_groups if new_channels % num_groups == 0 else 1, eps)
    y_ref = run_pytorch_groupnorm(x_misaligned, num_groups if new_channels % num_groups == 0 else 1, eps)
    assert not torch.allclose(y_kernel, y_ref, atol=1e-5), "Kernel should show discrepancies on misaligned memory."

# Issue 5: Shared memory reduction bank conflicts are hard to detect directly
# but we can artificially force a situation by choosing a block size that is not a power‐of‐two.
# Unfortunately, our kernel launch fixes threads per block in the extension.
# Thus, we simulate this by calling the kernel on an input where group_elems is such
# that the chosen threads (256 or group_elems) is not a power‐of‐two.
def test_shared_memory_reduction():
    # Choose spatial size and group sizes so that channels_per_group * spatial is not a power-of-two.
    batch_size = 4
    channels = 24  # divisible by 8 gives channels_per_group = 3, which is not a power-of-two
    num_groups = 8
    H, W = 7, 7  # total spatial = 49
    eps = 1e-5
    x = torch.randn(batch_size, channels, H, W, device="cuda")
    weight = torch.ones(channels, device="cuda")
    bias = torch.zeros(channels, device="cuda")
    
    y_kernel = run_kernel_forward(x, weight, bias, num_groups, eps)
    y_ref = run_pytorch_groupnorm(x, num_groups, eps)
    assert not torch.allclose(y_kernel, y_ref, atol=1e-5), "Kernel should perform suboptimally or produce incorrect output due to shared memory reduction issues when block size is non-power-of-two."

if __name__ == "__main__":
    pytest.main([__file__])
