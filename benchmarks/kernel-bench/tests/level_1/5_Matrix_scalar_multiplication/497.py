
import pytest
import torch
from torch.utils.cpp_extension import load
import os

def build_kernel():
    # Make sure the file 'kernel.cu' exists in the same directory as this test file.
    module = load(
        name="test_module",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_input_tensor_type():
    # Issue 2:
    # Pass a non-float32 (double) tensor to trigger the TORCH_CHECK error.
    module = build_kernel()
    # Create a double tensor and expect a runtime error.
    A = torch.randn(1024, 1024, dtype=torch.double, device="cuda")
    s = 3.14
    with pytest.raises(RuntimeError, match="Input tensor A must be of type float."):
        _ = module.forward(A, s)
    torch.cuda.synchronize()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_non_contiguous_tensor():
    # Issue 1:
    # Create a non-contiguous tensor by transposing the input.
    module = build_kernel()
    A = torch.randn(1024, 512, device="cuda", dtype=torch.float32)
    A_non_contig = A.t()  # This makes the tensor non-contiguous.
    s = 2.0
    # The custom kernel assumes contiguous memory, so the output may be incorrect.
    # We expect the kernel result to differ from the correct multiplication.
    C_kernel = module.forward(A_non_contig, s)
    C_expected = A_non_contig * s
    # We force synchronization to detect any possible kernel errors.
    torch.cuda.synchronize()
    # Check that the results do not match, indicating a problem.
    # (A correct implementation would handle non-contiguity or raise an error.)
    assert not torch.allclose(C_kernel, C_expected, atol=1e-5), (
        "Kernel unexpectedly produced the correct result for a non-contiguous tensor, "
        "which indicates it may be misinterpreting non-contiguous memory."
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_misaligned_memory():
    # Issue 1 (and related to vectorized float4 loads/stores):
    # Create a tensor that is contiguous overall but intentionally misaligned by taking a slice.
    module = build_kernel()
    # Allocate a 1D tensor with extra elements so that slicing produces a misaligned pointer.
    A_full = torch.randn(1025, device="cuda", dtype=torch.float32)
    # Take a slice that starts at index 1. This likely produces a pointer not aligned to 16 bytes.
    A_misaligned = A_full[1:]
    s = 1.5
    C_kernel = module.forward(A_misaligned, s)
    C_expected = A_misaligned * s
    torch.cuda.synchronize()
    # It is possible that the kernel returns an incorrect result (or even crashes) because of misalignment.
    # We check that the result is not equal to the expected value.
    assert not torch.allclose(C_kernel, C_expected, atol=1e-5), (
        "Kernel produced the correct result on a misaligned tensor slice, which is unexpected "
        "given that the kernel assumes 16-byte alignment for vectorized processing."
    )
