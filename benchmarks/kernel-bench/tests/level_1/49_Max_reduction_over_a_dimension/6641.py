
import torch
import pytest
from torch.utils.cpp_extension import load

# Helper function to compile the extension from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="max_reduce_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Trigger non-contiguous tensor issue.
# Create a non-contiguous input by transposing dimensions.
def test_non_contiguous_input():
    device = "cuda"
    # Create tensor of shape (batch_size, dim1, dim2)
    batch_size, dim1, dim2 = 16, 256, 256
    x = torch.randn(batch_size, dim1, dim2, device=device)
    # Make x non-contiguous by transposing two axes that affect the inner loop layout.
    x_noncontig = x.transpose(1, 2)
    
    # Reduction along dim=2 in the original tensor corresponds to reduction along dim=1
    # in the transposed tensor.
    # We use the kernel with the assumption of contiguity.
    kernel = build_kernel()
    # In our kernel call, the dimension provided is with respect to the original layout.
    # Here we intentionally pass a non-contiguous tensor and a reduction dim that is no longer contiguous.
    try:
        # This should produce a wrong result (or behave unexpectedly) because the kernel assumes contiguity.
        out_kernel = kernel.forward(x_noncontig, 1)
        out_ref = torch.max(x_noncontig, dim=1)[0]
    except Exception as e:
        pytest.skip("Kernel did not run due to non-contiguous memory error: " + str(e))
    
    # The outputs will likely be different; we assert that they are not close.
    assert not torch.allclose(out_kernel, out_ref, atol=1e-3), (
        "Kernel unexpectedly returned the correct result for non-contiguous input. "
        "This masks the underlying assumption in the kernel."
    )

# Test 2: Trigger potential __ldg issue for half precision.
def test_half_precision():
    device = "cuda"
    batch_size, dim1, dim2 = 16, 256, 256
    # Create a contiguous half-precision tensor.
    x_half = torch.randn(batch_size, dim1, dim2, device=device).half()
    
    kernel = build_kernel()
    try:
        out_kernel = kernel.forward(x_half, 1)
        out_ref = torch.max(x_half, dim=1)[0]
    except Exception as e:
        pytest.skip("Kernel failed with half precision input: " + str(e))
    
    # If __ldg is not operating correctly on __half, the result may differ.
    assert not torch.allclose(out_kernel.float(), out_ref.float(), atol=1e-2), (
        "Kernel returned nearly correct result for half precision input, "
        "which is unexpected given the __ldg issues with half data type."
    )

# Test 3: Trigger error due to invalid reduction dimension (lack of proper error checking).
def test_invalid_dimension():
    device = "cuda"
    batch_size, dim1, dim2 = 16, 256, 256
    x = torch.randn(batch_size, dim1, dim2, device=device)
    
    # Provide an out-of-bound dimension.
    invalid_dim = 5  # x.dim() is 3, so 5 is invalid.
    kernel = build_kernel()
    
    with pytest.raises(Exception):
        # Expect an exception because the reduction dimension is invalid.
        _ = kernel.forward(x, invalid_dim)
        
# Note: Depending on how the kernel extension is integrated with PyTorch,
# the error for an invalid dimension might be caught at the C++ level with an abort
# or an uncaught error. This test assumes that an exception is raised.
        
if __name__ == "__main__":
    pytest.main([__file__])
