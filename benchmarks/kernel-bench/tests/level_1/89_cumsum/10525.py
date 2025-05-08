
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="cumsum_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: inner_size greater than BLOCK_SIZE.
#
# In the kernel, inner_size is computed as the product of dimensions after the scanned one.
# Because the kernel only launches min(inner_size, BLOCK_SIZE) threads per block (using threadIdx.x)
# elements with indices >= BLOCK_SIZE in the inner_size dimension are never processed.
# We set up a test where inner_size is deliberately larger than BLOCK_SIZE.
def test_inner_size_exceeds_block_size():
    my_module = build_kernel()
    
    # We choose a tensor layout where scanning along dim=1.
    # Let tensor shape be (outer, stride, inner_size) with inner_size > BLOCK_SIZE (256). For example, inner_size = 300.
    outer = 2
    stride = 5  # scanned dimension length; corresponds to x.size(1)
    inner_size = 300  # > 256, so we expect issues in the kernel.
    
    # Construct tensor with shape (outer, stride, inner_size) in float32 on CUDA.
    x = torch.randn(outer, stride, inner_size, device="cuda", dtype=torch.float32)
    
    # Run the custom kernel forward.
    custom_out = my_module.forward(x, dim=1)
    
    # Compute the expected result using torch.cumsum
    expected_out = torch.cumsum(x, dim=1)
    
    # The results should differ because the kernel never computed the cumulative sum for 
    # inner indices beyond the thread block range.
    if torch.allclose(custom_out, expected_out):
        raise AssertionError("Test failed: The kernel output matches torch.cumsum despite inner_size exceeding BLOCK_SIZE. " +
                             "This indicates that the kernel is mistakenly processing all inner elements.")
    else:
        # Optionally print the maximum difference for debugging.
        max_diff = (custom_out - expected_out).abs().max().item()
        print("Detected difference for inner_size > BLOCK_SIZE. Maximum absolute difference:", max_diff)

# Issue 2: Data type incompatibility.
#
# The kernel only supports float32. Passing a tensor of a different dtype (e.g., float64)
# should lead to an error (or at least an incorrect operation). Here, we test that an error
# is raised when using an unsupported data type.
def test_invalid_dtype():
    my_module = build_kernel()
    
    # Create a tensor with float64 (double) dtype.
    x = torch.randn(10, 20, device="cuda", dtype=torch.float64)
    
    # The kernel expects float32. We expect a runtime error either from type conversion,
    # a segfault, or a failed CHECK_INPUT if extended. Here, we wrap the call in pytest.raises.
    with pytest.raises(Exception):
        _ = my_module.forward(x, dim=1)

# Issue 3: Unnecessary use of shared memory.
#
# Although this design issue may not always lead to an outright failure,
# it can be highlighted by noticing that the kernel simply uses shared memory
# to load one element per thread and then immediately uses it (with __ldg as a backup).
# In a well designed kernel, either shared memory should be used to perform
# more sophisticated data reuse or eliminated if it only adds overhead.
#
# For the purpose of testing, we create a simple test that checks whether the kernel runs
# and produces output. However, since performance or resource usage isn’t directly observable
# via correctness tests, here we only verify that the use of shared memory did not hide a synchronisation bug.
def test_shared_memory_usage():
    my_module = build_kernel()
    
    # Create a tensor with a typical shape for which inner_size is safe (< BLOCK_SIZE).
    x = torch.randn(64, 100, device="cuda", dtype=torch.float32)
    
    # Run the kernel; we are not comparing the result here as torch.cumsum would be correct,
    # but the purpose is to ensure that the kernel runs without deadlock.
    out = my_module.forward(x, dim=1)
    
    expected = torch.cumsum(x, dim=1)
    # Check that the results are approximately equal.
    assert torch.allclose(out, expected, atol=1e-4), "The kernel output is incorrect even in a safe shared memory scenario."
