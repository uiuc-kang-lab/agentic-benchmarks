
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    # Compile the CUDA extension from 'kernel.cu'.
    cuda_module = load(
        name="cumprod_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Non-contiguous tensor input should produce wrong results.
def test_non_contiguous_input():
    # Create a 2D tensor that is contiguous first, then make a non-contiguous view.
    N, M = 128, 4000
    # Create a tensor with shape (N, M)
    a = torch.randn(N, M, device="cuda", dtype=torch.float32)
    # Make it non-contiguous: for example, take the transpose.
    a_noncontig = a.t()  # shape becomes (M, N) and non-contiguous relative to original expectation.
    # We want to accumulate along dim=0 in the transposed view.
    dim = 0
    # Our custom kernel does not account for non-contiguity.
    kernel_module = build_kernel()
    output_kernel = kernel_module.forward(a_noncontig, dim)
    output_reference = torch.cumprod(a_noncontig, dim=dim)
    torch.cuda.synchronize()
    # This test is expected to fail -- if they match then the kernel would be handling non-contiguous input,
    # but the current indexing is likely wrong.
    assert not torch.allclose(output_kernel, output_reference, atol=1e-5), \
        "Kernel unexpectedly handled a non-contiguous input correctly. (It was expected to produce errors.)"

# Test 2: Multidimensional input (more than 2 dimensions) with cumulative product along a non-last dimension.
def test_multidimensional_input():
    # Create a 3D tensor; for contiguous tensor the kernel’s indexing (batch division etc.) is not general.
    a = torch.randn(4, 5, 6, device="cuda", dtype=torch.float32)
    # We pick dim=1 (the second dimension) for cumprod.
    dim = 1
    kernel_module = build_kernel()
    output_kernel = kernel_module.forward(a, dim)
    output_reference = torch.cumprod(a, dim=dim)
    torch.cuda.synchronize()
    # The outputs should differ because the kernel computes incorrect indexing for >2 dimensions.
    assert not torch.allclose(output_kernel, output_reference, atol=1e-5), \
        "Kernel output unexpectedly matched the reference for a multi-dimensional input. (Indexing issue not triggered.)"

# Test 3: Half precision input may be incorrectly initialized because of product = 1.
def test_half_precision_input():
    # Create a simple tensor in half precision.
    a = torch.randn(128, 4000, device="cuda", dtype=torch.float16)
    dim = 1
    kernel_module = build_kernel()
    output_kernel = kernel_module.forward(a, dim)
    output_reference = torch.cumprod(a, dim=dim)
    torch.cuda.synchronize()
    # The result might be off because the constant “1” might not be properly represented in half precision.
    # We expect a mismatch.
    assert not torch.allclose(output_kernel, output_reference, atol=1e-2), \
        "Kernel output unexpectedly matched the reference for half precision input. (Half init issue not triggered.)"

# Test 4: No error checking on CUDA operations may let silent failures go unnoticed.
# We trigger a situation that is likely to cause an error by passing an invalid dimension.
def test_invalid_dimension():
    a = torch.randn(128, 4000, device="cuda", dtype=torch.float32)
    invalid_dim = 5  # Out of bounds for a 2D tensor.
    kernel_module = build_kernel()
    # The behavior is undefined, but since there is no error checking the kernel may silently process and produce nonsense.
    # We compare with torch.cumprod that should raise an error.
    with pytest.raises(IndexError):
        _ = torch.cumprod(a, dim=invalid_dim)
    # If our kernel call did not raise an error, then the missing error checking is exposed.
    try:
        _ = kernel_module.forward(a, invalid_dim)
        # If no error was raised here, then we fail the test.
        pytest.fail("Kernel call with an invalid dimension did not raise an error (missing error handling).")
    except Exception:
        pass

if __name__ == "__main__":
    pytest.main([__file__])
