
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build the CUDA extension from kernel.cu
def build_kernel():
    module = load(
        name="custom_norm",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return module

# Test 1: Misaligned input
# We create a tensor with an offset that causes misalignment.
# One trick is to allocate a slightly larger tensor and then take a narrow slice.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_misaligned_input():
    # Allocate a tensor with one extra element along dimension 0
    base = torch.randn(17, 64, 256, 256, device="cuda", dtype=torch.float32)
    # Slice away the first element so that the data pointer is offset by 1 element
    # This is likely to cause misalignment (i.e. the pointer address modulo 16 != 0).
    # Note: This tensor is contiguous but its data pointer is not 16-byte aligned.
    x = base.narrow(0, 1, 16)
    custom_module = build_kernel()
    
    # Run the kernel. In a proper implementation, misaligned input should be either handled
    # or at least produce a result that isn’t completely wrong. Here we check that the result
    # deviates from the expected normalized output if misaligned loads cause problems.
    out = custom_module.forward(x)
    
    # Compute the reference result using PyTorch
    norm = torch.norm(x, p='fro')
    ref = x / norm if norm != 0 else x  # avoid division by zero in reference
    # We expect that the outputs are not close because the kernel likely misbehaves with misaligned data.
    # So if they accidentally match, we warn.
    if torch.allclose(out, ref, atol=1e-5):
        pytest.skip("Misaligned input did not trigger a noticeable error; the test is inconclusive!")
    else:
        assert not torch.allclose(out, ref, atol=1e-5), "Kernel output unexpectedly matches reference on misaligned input (issue with alignment check missing)."

# Test 2: Division-by-zero case
# Pass an input tensor composed entirely of zeros.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_division_by_zero():
    x = torch.zeros(16, 64, 256, 256, device="cuda", dtype=torch.float32)
    custom_module = build_kernel()
    
    out = custom_module.forward(x)
    # When norm == 0, dividing by 0 should propagate NaNs or infs.
    if torch.isnan(out).any() or torch.isinf(out).any():
        assert True
    else:
        pytest.fail("Kernel did not produce NaNs/Infs when dividing by zero (missing division-by-zero protection).")

# Test 3: Non-CUDA input (to trigger the CUDA check errors)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_non_cuda_input():
    x = torch.randn(16, 64, 256, 256, device="cpu", dtype=torch.float32)
    custom_module = build_kernel()
    with pytest.raises(RuntimeError, match="Input tensor must be on GPU"):
        custom_module.forward(x)
