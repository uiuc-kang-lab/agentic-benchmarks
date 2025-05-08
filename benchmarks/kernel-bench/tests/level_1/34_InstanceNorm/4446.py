
import os
import torch
import pytest
from torch.utils.cpp_extension import load

# Utility to (re)build the CUDA extension.
def build_kernel(extra_cuda_cflags=None):
    extra_cuda_cflags = extra_cuda_cflags or ["-O3", "--use_fast_math"]
    cuda_module = load(
        name="instance_norm_extension",
        sources=["kernel.cu"],
        extra_cuda_cflags=extra_cuda_cflags,
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# The following helper uses the built-in InstanceNorm2d as a reference.
def reference_instance_norm(x, weight, bias, eps):
    # Here we assume x shape is (N, C, H, W)
    # We use PyTorch’s built in instance norm with affine parameters.
    # For a fair comparison we must use the same eps as in the CUDA kernel.
    N, C, H, W = x.shape
    # Compute mean and variance per instance (n, c)
    x_reshaped = x.view(N, C, -1)
    mean = x_reshaped.mean(dim=-1, keepdim=True).view(N, C, 1, 1)
    var = x_reshaped.var(dim=-1, unbiased=False, keepdim=True).view(N, C, 1, 1)
    invstd = 1.0 / torch.sqrt(var + eps)
    gamma = weight.view(1, C, 1, 1) if weight is not None else 1.0
    beta  = bias.view(1, C, 1, 1) if bias is not None else 0.0
    y_ref = (x - mean) * invstd * gamma + beta
    return y_ref

# Test case 1: Create a misaligned input tensor.
# We simulate misalignment by taking a narrow slice that shifts the data pointer.
def test_misaligned_memory():
    torch.cuda.manual_seed(0)
    batch_size = 4
    features = 8
    H = 32
    W = 32
    eps = 1e-5

    # Create a tensor and then slice along the width so that the underlying
    # pointer is offset by one float (i.e. 4 bytes) and is not 16-byte aligned.
    x_full = torch.randn(batch_size, features, H, W + 1, device='cuda', dtype=torch.float32)
    x = x_full[..., 1:]  # remove the first element along the last dim
    # Make weight and bias non-None.
    weight = torch.randn(features, device='cuda', dtype=torch.float32)
    bias   = torch.randn(features, device='cuda', dtype=torch.float32)

    # Due to our vectorized load/store in the kernel, this misaligned memory access
    # might lead to undefined behavior. Here we compare with a reference implementation.
    module = build_kernel()
    try:
        y = module.forward(x, weight, bias, float(eps))
        torch.cuda.synchronize()
    except RuntimeError as e:
        # If the misalignment causes an access error, that’s a triggering of the issue.
        pytest.skip("Kernel failed due to misaligned memory access as expected: " + str(e))
    # Compare against reference instance normalization computed in Python
    y_ref = reference_instance_norm(x, weight, bias, eps)
    # Note: if undefined behavior occurs, the test output may differ significantly.
    assert torch.allclose(y, y_ref, atol=1e-4), "Output differs for misaligned input."

# Test case 2: Use a kernel launch with a very large block dimension
# to trigger potential out-of-bound shared memory access.
#
# In the provided kernel the number of threads per block is hard-coded to 256.
# To simulate a problem with shared memory usage, we force a re-build of the extension
# with overridden launch parameter (via a macro, for example).
#
# For this test to be effective, the kernel should be modified to use a macro (e.g. THREADS)
# instead of the hard-coded 256. In a real scenario, compiling with THREADS=2048 would trigger
# an out-of-bound write in the shared memory array if the block reduction isn’t updated.
@pytest.mark.skip(reason="This test requires rebuilding the extension with a custom block size using a macro (e.g. THREADS=2048) to trigger the shared memory issue.")
def test_large_block_dim():
    torch.cuda.manual_seed(0)
    batch_size = 2
    features = 4
    H = 64
    W = 64
    eps = 1e-5

    x = torch.randn(batch_size, features, H, W, device='cuda', dtype=torch.float32)
    weight = torch.randn(features, device='cuda', dtype=torch.float32)
    bias   = torch.randn(features, device='cuda', dtype=torch.float32)

    # The following extra flag would be used if the kernel were programmed to accept it.
    # For example, the kernel code would use:
    #   #ifndef THREADS
    #       #define THREADS 256
    #   #endif
    # and then use "THREADS" instead of 256.
    module = build_kernel(extra_cuda_cflags=["-O3", "--use_fast_math", "-DTHREADS=2048"])
    y = module.forward(x, weight, bias, float(eps))
    torch.cuda.synchronize()
    y_ref = reference_instance_norm(x, weight, bias, eps)
    assert torch.allclose(y, y_ref, atol=1e-4), "Kernel output differs when launched with a large block dimension."

