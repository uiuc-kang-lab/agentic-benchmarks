
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build the CUDA extension. It is assumed that "kernel.cu" is in the same directory.
def build_kernel():
    cuda_module = load(
        name="cumsum_kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.fixture(scope="module")
def cuda_module():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return build_kernel()

def test_input_tensor_type(cuda_module):
    # Issue 1: Pass a tensor with dtype float64. The kernel expects float32,
    # so this should trigger undefined behavior (or at least a check failure if type was enforced).
    x = torch.randn(128, 10, device="cuda", dtype=torch.float64)
    with pytest.raises(RuntimeError):
        # The kernel does not check type so it might not raise an error directly,
        # but the result will be incorrect. Here we expect a runtime error due to misinterpreting the memory.
        _ = cuda_module.forward(x, 1)
    # If it doesn't raise, we compare the output with torch.cumsum,
    # expecting a mismatch if the kernel misinterprets double data.
    # Uncomment below if you want to capture incorrect results instead of an exception.
    # out = cuda_module.forward(x, 1)
    # ref = torch.cumsum(x, dim=1)
    # assert not torch.allclose(out, ref), "Kernel incorrectly handled non-float32 input."

def test_non_contiguous_input(cuda_module):
    # Issue 2: Pass a non-contiguous tensor.
    # Create a contiguous tensor and then perform an operation to make it non-contiguous.
    x = torch.randn(128, 10, device="cuda", dtype=torch.float32)
    x_non_contig = x.t()  # Transpose makes the tensor non-contiguous.
    with pytest.raises(RuntimeError, match="must be contiguous"):
        _ = cuda_module.forward(x_non_contig, 0)

def test_warp_size_assumption(cuda_module):
    # Issue 3: The kernel hardcodes warp-scan behavior with warp size of 32.
    # We simulate this issue by selecting an input length (stride) that is <= 32 but not equal to warp size,
    # and then comparing the output with the PyTorch cumsum result.
    # If the warp-scan kernel incorrectly handles boundaries, the result will differ.
    stride = 13  # less than 32 and non-power-of-two
    x = torch.randn(128, stride, device="cuda", dtype=torch.float32)
    out_kernel = cuda_module.forward(x, 1)
    out_ref = torch.cumsum(x, dim=1)
    # The expectation is that they should be equal.
    # If the warp-scan kernel's hardcoded assumptions break on unexpected warp sizes or boundaries,
    # this test will fail.
    assert torch.allclose(out_kernel, out_ref, atol=1e-5), "Kernel warp-scan behavior incorrect for stride<32"
