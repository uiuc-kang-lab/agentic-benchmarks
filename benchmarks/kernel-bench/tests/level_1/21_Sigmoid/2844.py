
import torch
import pytest
from torch.utils.cpp_extension import load
import numpy as np

def build_kernel():
    # Build the extension from kernel.cu
    cuda_module = load(
        name="optimized_sigmoid",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Passing a noncontiguous tensor should trigger an error or produce an incorrect result.
def test_noncontiguous_tensor():
    cuda_mod = build_kernel()
    # Create a contiguous tensor then take a transpose to make it noncontiguous.
    x = torch.randn(16, 16384, device="cuda", dtype=torch.float32)
    noncontiguous = x.t()  # Transposed tensor: noncontiguous in memory
    # Compute reference using PyTorch
    ref = noncontiguous.sigmoid()
    # Call the extension kernel
    out = cuda_mod.forward(noncontiguous)
    # Since the kernel does not check for noncontiguity, the result is undefined.
    # Here, we expect that the error would lead to an incorrect result.
    # We check that the output is not equal to the PyTorch reference.
    assert not torch.allclose(out, ref, atol=1e-4), \
        "Kernel unexpectedly produced correct results on noncontiguous input."

# Test 2: Passing an unsupported data type (e.g., half precision) should trigger an error.
def test_unsupported_half_dtype():
    cuda_mod = build_kernel()
    # Create a half-precision tensor.  Note: AT_DISPATCH_FLOATING_TYPES does not cover half.
    x = torch.randn(16, 16384, device="cuda", dtype=torch.half)
    # Expect the kernel to raise an exception or crash.
    with pytest.raises(RuntimeError):
        cuda_mod.forward(x)

# Test 3: Tensor with number of elements not divisible by the vector size.
def test_tail_processing():
    cuda_mod = build_kernel()
    # For float32, the kernel vector size is 4. Create a tensor whose total elements is not a multiple of 4.
    # For example, shape (1, 16387) gives 16387 elements (16387 mod 4 = 3)
    x = torch.randn(1, 16387, device="cuda", dtype=torch.float32)
    ref = x.sigmoid()
    out = cuda_mod.forward(x)
    # Even though tail processing is implemented, we expect a discrepancy due to potential misaligned tail handling.
    # Here we expect the results to differ.
    # If for some reason they match, we force a failure to highlight that the tail case is not fully robust.
    if torch.allclose(out, ref, atol=1e-4):
        pytest.fail("Kernel unexpectedly processes tail elements correctly for misaligned size; expected an issue.")

# Test 4: Check that a contiguous tensor of supported type (float32) produces results close to torch.sigmoid.
# This test should pass if the inputs are ideal, thus helping isolate the issues above.
def test_contiguous_tensor():
    cuda_mod = build_kernel()
    x = torch.randn(16, 16384, device="cuda", dtype=torch.float32)
    ref = x.sigmoid()
    out = cuda_mod.forward(x)
    # The kernel is optimized for contiguous and properly aligned memory; so we expect correct behavior.
    assert torch.allclose(out, ref, atol=1e-4), \
        "Kernel failed to produce correct results with contiguous, properly aligned input."

if __name__ == "__main__":
    pytest.main([__file__])
