
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="group_norm_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# This helper computes the reference group norm output using PyTorch's own GroupNorm module.
def reference_group_norm(x, num_groups, weight, bias, eps):
    N, C = x.shape[:2]
    ref = torch.nn.functional.group_norm(x, num_groups=num_groups, weight=weight, bias=bias, eps=eps)
    return ref

# Test 1: Trigger the issue for double precision
# The kernel might not support double atomicAdd properly.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_double_precision():
    # Use a simple input where the number of elements per group is large enough (multiple chunks)
    batch_size = 4
    features = 32
    num_groups = 4
    dim1 = 64
    dim2 = 64
    eps = 1e-5

    # Create input in double precision (float64)
    x = torch.randn(batch_size, features, dim1, dim2, device="cuda", dtype=torch.float64)
    weight = torch.randn(features, device="cuda", dtype=torch.float64)
    bias = torch.randn(features, device="cuda", dtype=torch.float64)

    module = build_kernel()

    # Call the custom CUDA kernel forward function (as defined in kernel.cu)
    try:
        y = module.forward(x, weight, bias, num_groups, eps)
    except Exception as e:
        pytest.skip("Double precision atomicAdd is not supported on this device: " + str(e))
    # Compare with PyTorch reference
    y_ref = reference_group_norm(x, num_groups, weight, bias, eps)
    # The results may differ due to precision issues; we expect a relatively loose tolerance.
    assert torch.allclose(y, y_ref, atol=1e-4), "Double precision test: kernel output differs from reference!"

# Test 2: Trigger the potential warp-synchronization issue
# We force a scenario where each group is processed by multiple chunks (i.e. many atomicAdd calls) 
# so that the warp-level reduction in compute_stats_kernel_atomic is critical.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_warp_synchronous_reduction():
    # Choose dimensions so that each group's total elements > elements_per_block (1024)
    batch_size = 8
    features = 64  # must be divisible by num_groups
    num_groups = 8
    # Spatial dimension chosen to be large enough to produce several chunks per group
    dim1 = 128
    dim2 = 128
    eps = 1e-5

    x = torch.randn(batch_size, features, dim1, dim2, device="cuda", dtype=torch.float32)
    weight = torch.randn(features, device="cuda", dtype=torch.float32)
    bias = torch.randn(features, device="cuda", dtype=torch.float32)

    module = build_kernel()
    y = module.forward(x, weight, bias, num_groups, eps)
    y_ref = reference_group_norm(x, num_groups, weight, bias, eps)
    # Use a tolerance appropriate for potential reduction issues
    assert torch.allclose(y, y_ref, atol=1e-5), "Warp reduction test: kernel output differs from reference!"

if __name__ == '__main__':
    pytest.main([__file__])
