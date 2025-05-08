
import torch
import pytest
from torch.utils.cpp_extension import load
import os

# Build the CUDA kernel extension
def build_kernel():
    # Ensure the absolute path to the source file
    here = os.path.abspath(os.path.dirname(__file__))
    module = load(
        name="softsign_kernel_module",
        sources=[os.path.join(here, "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

# Expected softsign implementation (CPU reference)
def softsign_ref(x: torch.Tensor) -> torch.Tensor:
    return x / (1 + x.abs())

# Test 1: Use a tensor of type float64. Because the kernel always assumes float32,
# the output will be wrong (or may even crash) when the input type is wrong.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_wrong_dtype():
    kernel_module = build_kernel()
    # Create a double tensor. Kernel does not check type so it will reinterpret the data.
    x = torch.randn(1024, device="cuda", dtype=torch.float64)
    # Expectation: the wrong type leads to an output that does not match the reference softsign.
    with pytest.raises(AssertionError):
        y = kernel_module.forward(x)
        torch.cuda.synchronize()
        y_ref = softsign_ref(x)
        # The values should mismatch (or the operation may crash). We compare and force an assertion.
        assert torch.allclose(y, y_ref, atol=1e-5), "Kernel accepted non-float32 tensor!"

# Test 2: Create a tensor that is contiguous but deliberately misaligned for vectorized loads.
# One way to force misalignment is to create a tensor with extra elements and then take a narrow/slice view
# that does not start at the naturally aligned address.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_misaligned_tensor():
    kernel_module = build_kernel()
    # Allocate extra element so that slicing starting at index 1 may yield misaligned data.
    x_full = torch.randn(1025, device="cuda", dtype=torch.float32)
    # Create a misaligned view by slicing: this view might have its data_ptr offset by sizeof(float)
    x = x_full.narrow(0, 1, 1024)  # new tensor starting at element 1
    # Although x is contiguous, its underlying memory may not be aligned to 16 bytes.
    y = kernel_module.forward(x)
    torch.cuda.synchronize()
    y_ref = softsign_ref(x)
    # We expect either a wrong result (if misaligned loads produce garbage) 
    # or at least the difference should be noticeable.
    assert not torch.allclose(y, y_ref, atol=1e-5), "Kernel succeeded on a misaligned tensor unexpectedly!"

# Test 3: Lack of kernel launch error checking
# This test makes the kernel configuration intentionally "bad" by passing a tensor with zero elements.
# If the kernel did proper error checking after launch, it might catch an issue.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_empty_tensor():
    kernel_module = build_kernel()
    # Create an empty tensor (with float32 type and contiguous)
    x = torch.empty(0, device="cuda", dtype=torch.float32)
    # This is a boundary case; the kernel may run but no computation is needed.
    y = kernel_module.forward(x)
    torch.cuda.synchronize()
    y_ref = softsign_ref(x)
    # Check that the output is also empty – if kernel launch error handling was missing,
    # it might not properly handle 0 elements.
    assert y.numel() == 0 and torch.allclose(y, y_ref), "Kernel did not correctly handle empty tensor!"
