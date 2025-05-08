
import torch
import pytest
from torch.utils.cpp_extension import load

# Helper function to build and load the CUDA kernel module from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="logsoftmax_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True
    )
    return cuda_module

# Issue 1 & 2: Test with a dim_size smaller than the number of threads.
# For example, choose an input where the last dimension is 150.
def test_reduction_boundaries():
    # Create an input tensor with shape (batch_size, dim_size) where
    # dim_size is not a power of two so that next_power_of_two > dim_size.
    batch_size = 8
    dim_size = 150  # Not a power of two. next_power_of_two(150)=256 on most systems.
    # Use random values; the error in reduction may produce wrong results (or NaNs)
    x = torch.randn(batch_size, dim_size, device="cuda", dtype=torch.float32)
    
    kernel_module = build_kernel()
    # Since our kernel expects the dimension on the last axis to be our softmax dim,
    # pass dim=-1 (or dim=1 if no permutation occurs).
    y = kernel_module.forward(x, -1)
    # Compute reference using PyTorch's log_softmax.
    y_ref = torch.log_softmax(x, dim=-1)
    
    # The error might produce a result that is not close.
    # We test for closeness; we expect failure if the issue is present.
    assert not torch.allclose(y, y_ref, atol=1e-4), "Reduction boundary issue not triggered as expected."

# Issue 3: Test with dim_size = 1 to trigger undefined behavior in next_power_of_two.
def test_dim_size_one():
    batch_size = 8
    dim_size = 1
    x = torch.randn(batch_size, dim_size, device="cuda", dtype=torch.float32)
    
    kernel_module = build_kernel()
    # Using the only valid dimension (-1) so that the kernel works on the last dimension.
    y = kernel_module.forward(x, -1)
    y_ref = torch.log_softmax(x, dim=-1)
    
    # Compare outputs. Due to undefined behavior, we may get a wrong result.
    # We assert that the outputs differ significantly.
    max_diff = (y - y_ref).abs().max().item()
    assert max_diff > 1e-3, f"Expected significant difference for dim_size=1, but got max diff = {max_diff}"

# Issue 4 (Minor): Building with missing proper qualification of max may lead to compile warnings
# or errors on some systems. We simulate a run just to ensure the module loads.
def test_module_load():
    try:
        _ = build_kernel()
    except Exception as e:
        pytest.fail(f"Kernel module failed to load due to missing includes or other issues: {e}")
    
if __name__ == "__main__":
    pytest.main([__file__])
