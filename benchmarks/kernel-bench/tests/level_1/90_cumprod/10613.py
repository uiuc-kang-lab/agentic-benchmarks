
import pytest
import torch
from torch.utils.cpp_extension import load

# Utility function to compile and load the CUDA extension.
def build_kernel():
    cuda_module = load(
        name="cumprod_kernel_ext",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Test 1: Trigger issue with incorrect index computations in higher dimensional tensors.
def test_incorrect_indexing_with_3d():
    # Create a 3D tensor. The kernel assumes a 2D contiguous layout.
    x = torch.randn(4, 5, 6, device="cuda")
    # Choose a cumulative dimension that is not the last one.
    dim = 1
    torch_ref = torch.cumprod(x, dim=dim)
    
    kernel = build_kernel()
    # The kernel expects the cumulative dim to be mapped via: 
    # total_threads = numel / sizes[dim] --- which is valid only for a 2D layout.
    y = kernel.forward(x, dim)
    torch.cuda.synchronize()
    # Here we expect a mismatch since the indexing is not general.
    assert not torch.allclose(y, torch_ref), "Test failed: Kernel unexpectedly matched torch.cumprod output for a 3D tensor."

# Test 2: Trigger issue with non-contiguous tensor.
def test_non_contiguous_tensor():
    # Create a contiguous tensor then transpose to get a non-contiguous one.
    x = torch.randn(128, 4000, device="cuda")
    x_noncontig = x.t()  # Now shape is (4000, 128) and non contiguous
    # Choose cumulative dimension accordingly.
    dim = 0  # We want to cumprod over rows of the transposed tensor.
    torch_ref = torch.cumprod(x_noncontig, dim=dim)
    
    kernel = build_kernel()
    y = kernel.forward(x_noncontig, dim)
    torch.cuda.synchronize()
    # The computed cumulative product will likely be wrong due to stride assumptions.
    assert not torch.allclose(y, torch_ref), "Test failed: Kernel unexpectedly produced correct output for non-contiguous input."

# Test 3: Trigger issue with unsupported input tensor type.
def test_unsupported_dtype():
    # Create an integer tensor, which is not supported by the AT_DISPATCH_FLOATING_TYPES_AND_HALF macro.
    x = torch.randint(1, 10, (128, 4000), device="cuda", dtype=torch.int32)
    dim = 1
    kernel = build_kernel()
    with pytest.raises(RuntimeError):
        # Expect a runtime error or dispatch error because int32 is unsupported.
        _ = kernel.forward(x, dim)

# Test 4: Trigger potential issues due to loop unrolling when dim_size is large.
def test_large_dim_size_unroll():
    # Create a tensor with a very large cumulative dimension,
    # which might trigger issues with the #pragma unroll.
    x = torch.randn(64, 10000, device="cuda")
    dim = 1
    torch_ref = torch.cumprod(x, dim=dim)
    
    kernel = build_kernel()
    y = kernel.forward(x, dim)
    torch.cuda.synchronize()
    # We expect difference because unrolling a large loop with a runtime bound may fail.
    assert not torch.allclose(y, torch_ref, atol=1e-5), "Test failed: Kernel unexpectedly produced correct result for large dim_size (unrolling issue not triggered)."

# Test 5: Trigger lack of error checking by forcing an obviously incorrect tensor shape.
def test_empty_tensor():
    # Passing an empty tensor may cause out-of-bound accesses and expose the lack of error checking.
    x = torch.tensor([], device="cuda", dtype=torch.float32)
    dim = 0
    kernel = build_kernel()
    # In this case, the behavior is undefined; we check that the kernel does not quietly produce a valid result.
    y = kernel.forward(x, dim)
    torch.cuda.synchronize()
    # For an empty tensor, torch.cumprod returns an empty tensor.
    torch_ref = torch.cumprod(x, dim=dim)
    # We expect the results to be different (or the kernel should have raised an error if error checking were present).
    assert not torch.equal(y, torch_ref), "Test failed: Kernel produced a valid result for an empty tensor despite lack of error checking."
    
if __name__ == "__main__":
    pytest.main([__file__])
