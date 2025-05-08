
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build the CUDA extension from kernel.cu
def build_kernel():
    # Assume kernel.cu is in the same directory as this test file.
    src_path = os.path.join(os.path.dirname(__file__), "kernel.cu")
    cuda_module = load(
        name="kernel_module",
        sources=[src_path],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.fixture(scope="module")
def kernel_module():
    return build_kernel()

def test_non_contiguous_tensor(kernel_module):
    # Issue 1: Test with a non-contiguous input tensor.
    # Create a contiguous tensor and then form a non-contiguous view from it.
    x = torch.randn(32, 32, device="cuda", dtype=torch.float32)
    non_contig = x.t()  # Transposed, non-contiguous.
    # Compute reference using torch.tanh on a contiguous copy.
    ref = torch.tanh(non_contig.contiguous()).view(non_contig.size())
    # Pass non_contiguous tensor to kernel.
    out = kernel_module.forward(non_contig)
    torch.cuda.synchronize()
    # The output might be wrong due to non-contiguity.
    # This test is designed to detect the issue.
    assert not torch.allclose(out, ref, atol=1e-5), (
        "Test failed to trigger non-contiguity issue: kernel handled non-contiguous input unexpectedly."
    )

def test_alignment_issue(kernel_module):
    # Issue 2: Test with an input tensor that is likely misaligned.
    # It is difficult to control memory alignment in PyTorch.
    # However, slicing can produce a view that is not aligned to a float4 boundary.
    # We allocate a slightly larger tensor and then slice an offset view.
    base = torch.randn(1025, device="cuda", dtype=torch.float32)
    # Slicing starting from index 1 might break the alignment for vectorized load.
    misaligned = base.narrow(0, 1, 1024)
    ref = torch.tanh(misaligned.contiguous())
    out = kernel_module.forward(misaligned)
    torch.cuda.synchronize()
    # We expect the outputs to differ because of misaligned loads in the specialized kernel.
    assert not torch.allclose(out, ref, atol=1e-5), (
        "Test failed to trigger alignment issue: kernel handled misaligned input unexpectedly."
    )

def test_half_precision_input(kernel_module):
    # Issue 3: Test with half precision input.
    # The kernel dispatch macros used in the extension do not cover float16.
    # We expect the module to either raise an error or produce an incorrect result.
    x = torch.randn(1024, device="cuda", dtype=torch.float16)
    with pytest.raises(Exception):
        # Expecting an error because half is not dispatched in the kernel.
        _ = kernel_module.forward(x)
