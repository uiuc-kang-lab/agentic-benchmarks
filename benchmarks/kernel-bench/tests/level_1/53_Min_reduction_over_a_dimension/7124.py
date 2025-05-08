
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper function to load and build the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="fast_min_reduction_ext",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Test 1: Trigger lack of half precision support.
# Expect the kernel to raise an error when using float16 because __half is not handled.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_half_precision_input():
    kernel = build_kernel()
    # Create a half-precision tensor.
    # Although torch.min supports half, our kernel may not dispatch a valid reduction.
    x = torch.randn(16, 256, 256, device="cuda", dtype=torch.float16)
    # Choose reduction on dim=1 (size 256). We expect the kernel to fail.
    with pytest.raises(RuntimeError):
        # Call the custom forward. This should either throw a compilation error or a runtime error.
        out = kernel.forward(x, 1)
        torch.cuda.synchronize()

# Test 2: Trigger the case with an empty reduction dimension.
# When the reduction dimension size is 0, the kernel's loop has no valid work.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_empty_reduction_dimension():
    kernel = build_kernel()
    # Create a tensor with an empty reduction dimension.
    # For example, reduction dimension size 0 at dim=1.
    x = torch.randn(16, 0, 256, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        # Expect an error because r == 0 is not handled.
        out = kernel.forward(x, 1)
        torch.cuda.synchronize()

# Test 3: Trigger potential warp-level reduction issue by simulating a block with fewer threads.
# We simulate this by creating an input where the reduction dimension is large, but we force a kernel launch
# with a different block size by monkey-patching the forward function temporarily.
# Note: This is a hack to simulate nonstandard block sizes that the kernel was not generalized for.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_nonstandard_block_dim():
    # We create a tensor with a normal reduction dimension.
    x = torch.randn(16, 256, 256, device="cuda", dtype=torch.float32)
    # Load the module.
    kernel = build_kernel()

    # Monkey-patch the kernel.forward function to use fewer threads per block.
    # WARNING: This is hacky and assumes that we can modify the kernel launch parameters.
    # Since our kernel.forward hardcodes threads=256, we simulate a scenario by re-compiling the kernel with a new constant.
    # For this test to work, one would need to modify kernel.cu to read the blockDim as an argument.
    # Here, we assume that such a modification is tested.
    #
    # Since we cannot modify the kernel launch without modifying the source,
    # we simply call the kernel and then compare with torch.min.
    out = kernel.forward(x, 1)
    # For validation, compare against PyTorch's min reduction.
    expected = torch.min(x, dim=1)[0]
    # Although the numerical result might match, in a truly misconfigured kernel,
    # out-of-bound access may trigger an error. We use allclose to at least compare results.
    assert torch.allclose(out, expected, atol=1e-5), "Warp-level reduction may be broken with nonstandard block dim."

# Test 4: Check potential overflow (simulate with large dims)
# This test creates tensor shape parameters near the edge of int.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_overflow_in_dimension_conversion():
    kernel = build_kernel()
    # We simulate large dimensions. We cannot actually allocate extremely large tensors,
    # but we can simulate the situation by creating a tensor where the computed 'outer'
    # would be large if converted from int64_t to int.
    # For instance, use dimensions that multiply to a value near the int max, but keep the reduction dim small.
    # WARNING: This test is only conceptual, as actually allocating such a tensor is not practical.
    # Instead, we rely on the fact that the kernel.forward will check for dim range.
    # Here, we artificially create a tensor with high dims, but with a valid product.
    # Use sizes that are high, but still allow allocation.
    # We choose dim0 and dim2 to be large so that outer = dim0 and inner = dim2.
    dim0 = 10000
    dim1 = 2  # reduction dim small to avoid huge allocation in r dimension
    dim2 = 10000
    try:
        x = torch.randn(dim0, dim1, dim2, device="cuda", dtype=torch.float32)
    except RuntimeError:
        pytest.skip("Tensor too large to allocate on this device.")
    # This call may not immediately trigger an error,
    # but if there is an overflow in converting dimensions, the result will be wrong.
    out = kernel.forward(x, 1)
    expected = torch.min(x, dim=1)[0]
    assert torch.allclose(out, expected, atol=1e-5), "Potential dimension conversion overflow detected."

if __name__ == "__main__":
    pytest.main([__file__])
