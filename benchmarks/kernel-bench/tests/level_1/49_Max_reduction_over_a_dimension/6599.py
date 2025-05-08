
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1: Constant memory race condition with concurrent kernel launches.
def test_concurrent_kernel_launches():
    module = build_kernel()
    # Create an input tensor of shape (batch, dim1, dim2)
    x = torch.randn(8, 16, 32, device="cuda")
    
    # Launch two kernel invocations concurrently on different streams with different reduction dimensions.
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    result1 = None
    result2 = None

    # Use different reduction dimensions concurrently.
    with torch.cuda.stream(stream1):
        result1 = module.forward(x, 1)  # reduce along dim 1
    with torch.cuda.stream(stream2):
        result2 = module.forward(x, 2)  # reduce along dim 2

    torch.cuda.synchronize()

    # Compute reference outputs
    ref1 = torch.max(x, dim=1)[0]
    ref2 = torch.max(x, dim=2)[0]

    # Because of the shared constant memory for c_dim_size, one or both results may be wrong.
    # The test expects at least one of the two outputs to deviate from its reference.
    correct1 = torch.allclose(result1, ref1, atol=1e-4)
    correct2 = torch.allclose(result2, ref2, atol=1e-4)
    if correct1 and correct2:
        raise AssertionError("Concurrent kernel launches produced correct results unexpectedly. "
                             "This indicates that the constant memory (c_dim_size) is not causing a race condition.")

# Issue 2: Kernel assumes contiguous input.
def test_non_contiguous_input():
    module = build_kernel()
    # Create a contiguous input tensor then create a non-contiguous view by transposing two dimensions.
    x = torch.randn(4, 5, 6, device="cuda")
    x_nc = x.transpose(1, 2)  # non-contiguous view
    # Choose a reduction dimension that corresponds to the expected contiguous dimension in the kernel.
    # Here, if we reduce over dim=1 in x_nc, the kernel will think inner_size = size(2) from x_nc,
    # but the underlying memory layout is not as expected.
    dim_to_reduce = 1
    result = module.forward(x_nc, dim_to_reduce)
    # Compute the reference using torch.max which handles non-contiguous tensors.
    ref = torch.max(x_nc, dim=dim_to_reduce)[0]
    # We expect the kernel result to be different due to incorrect indexing.
    if torch.allclose(result, ref, atol=1e-4):
        raise AssertionError("Kernel unexpectedly produced correct results with non-contiguous input, "
                             "despite assuming contiguous memory.")

# Issue 3: Integer overflow in indexing for large tensors.
def test_large_tensor_index_overflow():
    # This test is designed to simulate the potential overflow issue.
    # Allocating a tensor so huge that outer_size * inner_size > INT_MAX is not practical;
    # instead, we simulate the condition by creating a tensor with moderately large outer and inner sizes.
    # Note: This test may be skipped if sufficient GPU memory is not available.
    pytest.skip("Skipping large tensor test due to memory constraints. "
                "This issue requires extremely large tensors that may not be allocatable in the test environment.")
    module = build_kernel()
    # Example: shape chosen so that outer_size * inner_size is huge.
    # Suppose we reduce over dim=1, then outer_size = x.size(0) is huge.
    x = torch.randn(300000, 10000, 10, device="cuda")  # unrealistic size; simulation only.
    result = module.forward(x, 1)
    ref = torch.max(x, dim=1)[0]
    if not torch.allclose(result, ref, atol=1e-4):
        raise AssertionError("Kernel result differs from reference for a huge tensor, "
                             "possibly due to integer overflow in thread indexing.")

# Issue 4: Kernel does not support integer tensor types.
def test_integer_dtype_input():
    module = build_kernel()
    # Create an input tensor with integer type.
    x = torch.randint(0, 100, (4, 5, 6), device="cuda", dtype=torch.int)
    with pytest.raises(RuntimeError):
        # Expect that the kernel dispatch fails because integer types are not handled.
        module.forward(x, 1)
        
if __name__ == "__main__":
    pytest.main([__file__])
