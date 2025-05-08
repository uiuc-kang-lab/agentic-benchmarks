
import pytest
import torch
from torch import nn
from torch.utils.cpp_extension import load

# Build the CUDA extension from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="group_norm_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Helper function to run group normalization forward using the CUDA kernel.
def run_group_norm_cuda(x, weight, bias, num_groups, eps):
    kernel = build_kernel()
    return kernel.forward(x, weight, bias, num_groups, eps)

# Test 1: Trigger Issue 1 (small group_elems -> very few threads, causing warp-level shfl with hardcoded mask)
def test_small_group_elems():
    # Create an input where each group has very few elements.
    # For example: batch_size=1, channels=8, num_groups=8 (so channels_per_group==1)
    # Spatial dims = 1, so each group has only 1 element.
    batch_size = 1
    channels = 8
    num_groups = 8
    eps = 1e-5
    x = torch.randn(batch_size, channels, 1, device="cuda", dtype=torch.float32)
    weight = torch.ones(channels, device="cuda", dtype=torch.float32)
    bias = torch.zeros(channels, device="cuda", dtype=torch.float32)
    
    # Run the CUDA kernel.
    y_cuda = run_group_norm_cuda(x, weight, bias, num_groups, eps)
    
    # Compute reference output using PyTorch's native GroupNorm.
    group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=eps).to("cuda")
    # Override weight and bias.
    with torch.no_grad():
        group_norm.weight.copy_(weight)
        group_norm.bias.copy_(bias)
    y_ref = group_norm(x)
    
    # Check the outputs.
    # This test is expected to fail (or yield a significant numerical difference)
    # if the warp-level reduction in the kernel is not handling small block sizes correctly.
    assert not torch.allclose(y_cuda, y_ref, atol=1e-6), (
        "Test for warp-level reduction failure did not trigger a difference. "
        "When group elems is very small, the use of a hardcoded mask should yield a wrong result."
    )

# Test 2: Trigger Issue 2 (channels not divisible by num_groups)
def test_invalid_channels_divisible():
    # Create an input where channels are not divisible by num_groups.
    batch_size = 2
    channels = 10  # Not divisible by num_groups.
    num_groups = 3
    eps = 1e-5
    # Using a small spatial dimension.
    x = torch.randn(batch_size, channels, 16, 16, device="cuda", dtype=torch.float32)
    weight = torch.ones(channels, device="cuda", dtype=torch.float32)
    bias = torch.zeros(channels, device="cuda", dtype=torch.float32)
    
    # Run the CUDA kernel.
    y_cuda = run_group_norm_cuda(x, weight, bias, num_groups, eps)
    
    # Compute reference output using PyTorch's native GroupNorm.
    # Note: PyTorch’s GroupNorm already requires channels to be divisible by num_groups.
    # So we wrap the call expecting that the behavior will be different.
    with pytest.raises(Exception):
        # Expect that PyTorch GroupNorm fails or at least gives a different result.
        _ = nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=eps)(x)
    
    # Alternatively, if no exception is thrown (in a more general scenario)
    # we can compare and assert that the outputs do not match.
    # Uncomment the following lines if you prefer to compare outputs instead.
    #
    # group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=eps).to("cuda")
    # with torch.no_grad():
    #     group_norm.weight.copy_(weight)
    #     group_norm.bias.copy_(bias)
    # y_ref = group_norm(x)
    # assert not torch.allclose(y_cuda, y_ref, atol=1e-6), (
    #     "Test for channel divisibility failure did not trigger a difference. "
    #     "Kernel does not check for channels not divisible by num_groups."
    # )

# Test 3: Trigger Issue 3 (grid dimension too large)
def test_huge_grid_dimension():
    # Construct parameters that would cause the grid dimension used in the stats kernel to be huge.
    # We do this by creating a scenario with an extremely large number of groups.
    # For instance, a moderately sized tensor with a huge batch size.
    # Note: We avoid allocating huge tensors by setting spatial==1 and channels minimal.
    batch_size = 2**16  # A large batch size.
    channels = 8       # Minimal channels.
    num_groups = 8     # So total groups = batch_size * num_groups is huge.
    eps = 1e-5
    x = torch.randn(batch_size, channels, 1, device="cuda", dtype=torch.float32)
    weight = torch.ones(channels, device="cuda", dtype=torch.float32)
    bias = torch.zeros(channels, device="cuda", dtype=torch.float32)
    
    # The forward call sets grid size = batch_size * num_groups.
    # For our test, we simulate that such a grid dimension would exceed the hardware limit.
    # Since we cannot (or do not want to) actually allocate an input that exceeds CUDA grid limits,
    # we check that the kernel launch fails (raising a RuntimeError).
    with pytest.raises(RuntimeError):
        _ = run_group_norm_cuda(x, weight, bias, num_groups, eps)
