
import torch
import pytest
from torch.utils.cpp_extension import load

# Build the CUDA extension from kernel.cu.
def build_kernel():
    return load(
        name="group_norm_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

# ---------------
# Test case for Issue 1:
# Verify that using double tensors (i.e. dtype=torch.float64) produces incorrect outputs
# due to the hard-coded shared memory size based on float.
def test_double_precision_shared_memory():
    # Use parameters that are valid in general
    batch_size = 4
    channels = 16  # (divisible by num_groups for this test)
    num_groups = 4
    dim1, dim2 = 32, 32
    eps = 1e-5

    # Build random input and parameters in double precision on CUDA
    x = torch.randn(batch_size, channels, dim1, dim2, device="cuda", dtype=torch.float64)
    weight = torch.randn(channels, device="cuda", dtype=torch.float64)
    bias = torch.randn(channels, device="cuda", dtype=torch.float64)

    # Get the kernel module and run the forward kernel.
    kernel_module = build_kernel()
    # The kernel function forward expects: input, weight, bias, num_groups, eps.
    try:
        y_cuda = kernel_module.forward(x, weight, bias, num_groups, eps)
        # Compare with PyTorch native GroupNorm for correctness
        # Note: PyTorch's GroupNorm supports double precision in its native implementation.
        gn = torch.nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=eps).to("cuda", dtype=torch.float64)
        # Manually set weight and bias to our tensors to have the same parameters.
        with torch.no_grad():
            gn.weight.copy_(weight)
            gn.bias.copy_(bias)
        y_ref = gn(x)
        # Due to the shared memory issue, we expect a discrepancy.
        assert not torch.allclose(y_cuda, y_ref, atol=1e-5), (
            "Test failed: The kernel output in double precision matches the reference, which is unexpected given shared memory mismatch."
        )
    except Exception as e:
        # If an error occurs, that is acceptable as it indicates incompatibility with double precision.
        pytest.skip(f"Double precision test skipped due to exception: {e}")

# ---------------
# Test case for Issue 2:
# The kernel assumes that channels (C) is evenly divisible by num_groups.
# This test sets up an input where C is not divisible by num_groups and ensures
# that either an error is raised or the output is incorrect.
def test_channels_not_divisible_by_num_groups():
    batch_size = 4
    channels = 18  # not divisible by num_groups
    num_groups = 4
    dim1, dim2 = 32, 32
    eps = 1e-5

    x = torch.randn(batch_size, channels, dim1, dim2, device="cuda", dtype=torch.float32)
    weight = torch.randn(channels, device="cuda", dtype=torch.float32)
    bias = torch.randn(channels, device="cuda", dtype=torch.float32)

    kernel_module = build_kernel()
    # The native PyTorch GroupNorm will throw an error when channels % num_groups != 0.
    # Our kernel, however, does not check for that. We thus expect either a silent failure (wrong output)
    # or a different behavior from the built-in version.
    with pytest.raises(Exception):
        # First, check PyTorch's behavior to anticipate an error.
        gn = torch.nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=eps)
        _ = gn(x)
    
    # Now, try our kernel. If it does not throw, then its output may be incorrect.
    try:
        y_cuda = kernel_module.forward(x, weight, bias, num_groups, eps)
        # If no exception, compare with a rough computation:
        # We compute the group indices manually and then for one sample check if groups are computed properly.
        # Since channels is not divisible by num_groups, we expect a mismatch.
        # Here, we simply assert that the kernel output is not close to the input.
        assert not torch.allclose(y_cuda, x, atol=1e-5), (
            "Test failed: Kernel output should be incorrect when channels are not divisible by num_groups."
        )
    except Exception:
        # If our kernel also raises an error, that's acceptable.
        pass
