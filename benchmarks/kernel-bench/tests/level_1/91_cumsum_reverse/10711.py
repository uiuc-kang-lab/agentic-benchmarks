
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="reverse_cumsum_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

def reference_reverse_cumsum(x, dim):
    return x.flip(dim).cumsum(dim).flip(dim)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_non_float_input():
    """
    Issue 1: The kernel is hardcoded for float32.
    Using a float64 tensor should trigger the fallback.
    """
    module = build_kernel()
    x = torch.randn(128, 32, device="cuda", dtype=torch.float64)
    dim = 1  # last dimension
    result = module.forward(x, dim)
    ref = reference_reverse_cumsum(x, dim)
    assert torch.allclose(result, ref, atol=1e-5), "Non float32 input did not fall back correctly."

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_non_last_dimension():
    """
    Issue 1: The kernel is implemented only for the last dimension.
    Using a cumulative sum along a non-last dimension should trigger the fallback.
    """
    module = build_kernel()
    x = torch.randn(64, 32, device="cuda", dtype=torch.float32)
    dim = 0  # not the last dimension (last dim is 1)
    result = module.forward(x, dim)
    ref = reference_reverse_cumsum(x, dim)
    assert torch.allclose(result, ref, atol=1e-5), "Non-last dimension input did not fall back correctly."

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_large_row_size():
    """
    Issue 2: The kernel only handles row sizes up to WARP_SIZE (32).
    Using a tensor with a last dimension larger than 32 should force a fallback.
    """
    module = build_kernel()
    x = torch.randn(32, 64, device="cuda", dtype=torch.float32)  # 64 > 32 (WARP_SIZE)
    dim = 1  # last dimension
    result = module.forward(x, dim)
    ref = reference_reverse_cumsum(x, dim)
    assert torch.allclose(result, ref, atol=1e-5), "Large row size input did not fall back correctly."

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_empty_last_dimension():
    """
    Issue 3: The kernel does not handle the case when the last dimension is empty.
    This should result in an error (e.g. division by zero or invalid kernel launch).
    """
    module = build_kernel()
    x = torch.randn(10, 0, device="cuda", dtype=torch.float32)  # empty last dimension
    dim = 1  # last dimension
    with pytest.raises(Exception):
        # Expecting an error because row_size is 0 which will lead to undefined behavior.
        module.forward(x, dim)
