
import torch
import pytest
from torch.utils.cpp_extension import load

# Helper function to compile and load the CUDA extension
def build_kernel():
    cuda_module = load(
        name="reverse_cumsum_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Helper: reference implementation using PyTorch operations
def ref_reverse_cumsum(x, dim):
    return torch.cumsum(x.flip(dim), dim=dim).flip(dim)

# Test case 1:
# This test uses a simple float tensor with a small last-dimension
# so that the specialized CUDA kernel is chosen. However, due to the
# double reversal error the outputs are incorrect.
def test_incorrect_reverse_cumsum():
    # Use a small 1x4 tensor with predetermined values
    # For input row = [1, 2, 3, 4], the intended reverse cumsum is:
    # flip: [4, 3, 2, 1] → cumsum: [4, 7, 9, 10] → flip: [10, 9, 7, 4]
    expected = torch.tensor([[10., 9., 7., 4.]], device="cuda")
    # But our kernel will process the row with reversed indexing and warp-scan,
    # which (for example) yields: [1, 3, 6, 9] (i.e. accumulation starting
    # from the wrong end).
    x = torch.tensor([[1., 2., 3., 4.]], device="cuda")
    # Force the kernel to be used: type is float,
    # op dimension is last, and size <= WARP_SIZE (4 <= 32).
    mod = build_kernel()
    out = mod.forward(x, 1)
    torch.cuda.synchronize()
    # The result should be different from what our kernel produces.
    assert not torch.allclose(out, expected), \
        f"Kernel output unexpectedly matched the correct result: {out}"
    # Also, report the erroneous output for debugging.
    print(f"Test 1 (incorrect accumulation) detected wrong output: {out}")

# Test case 2:
# This test provides input that does not satisfy the kernel’s strict conditions:
# for example, using a non-last dimension (dim != x.dim()-1) forces the fallback path.
# The fallback (using PyTorch’s native operations) computes the correct result.
def test_kernel_limitation_non_last_dimension():
    # Create a tensor of shape (4, 5) and choose to reduce along dim 0 (not last dim).
    x = torch.randn(4, 5, device="cuda")
    mod = build_kernel()
    # Since dim != x.dim()-1, our function bypasses the CUDA kernel and uses the fallback.
    out = mod.forward(x, 0)
    torch.cuda.synchronize()
    expected = ref_reverse_cumsum(x, 0)
    # The result from fallback should be correct
    assert torch.allclose(out, expected), \
        f"Fallback output wrong for non-last dimension usage: got {out}, expected {expected}"

if __name__ == "__main__":
    pytest.main([__file__])
