
import torch
import pytest
from torch.utils.cpp_extension import load
import os

# Build the CUDA kernel extension (assumes kernel.cu is in the same directory)
def build_kernel():
    return load(
        name="test_cumprod_module",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )

@pytest.fixture(scope="module")
def kernel_module():
    return build_kernel()

def test_non_contiguous_input(kernel_module):
    # Issue 1: Test with a non-contiguous tensor.
    # Create a contiguous tensor then make it non-contiguous by transposing.
    x = torch.randn(32, 64, device="cuda", dtype=torch.float32)
    # Transpose to force non-contiguity.
    x_non_contig = x.t()
    # For torch.cumprod, we need to choose a dimension. Here, choose dim=1.
    # Note: In the non-contiguous layout x_non_contig, even if the user asks for cumprod along dim=1,
    # the kernel indexing (which assumes contiguous layout) will likely produce the wrong result.
    ref = torch.cumprod(x_non_contig, dim=1)
    out = kernel_module.forward(x_non_contig, 1)
    torch.cuda.synchronize()
    # Check that the output differs from the reference due to index miscomputation.
    # The test is considered to trigger the issue if the outputs are not equal.
    assert not torch.allclose(out, ref, atol=1e-5), "Kernel incorrectly handled non-contiguous input!"

def test_wrong_dim_blocking(kernel_module):
    # Issue 2 & 3: Test with a tensor where the cumulative dimension is not in the "expected" position.
    # Consider a 3D tensor and perform cumprod along dim=0.
    # In a contiguous tensor of shape (A, B, C), strides may not match the assumed access pattern.
    A, B, C = 16, 32, 24
    x = torch.randn(A, B, C, device="cuda", dtype=torch.float32)
    # Choosing cumprod along dim=0.
    ref = torch.cumprod(x, dim=0)
    out = kernel_module.forward(x, 0)
    torch.cuda.synchronize()
    # Given the flawed total_batches calculation and index arithmetic, the output likely is incorrect.
    assert not torch.allclose(out, ref, atol=1e-5), "Kernel incorrectly computed cumulative product along dimension 0 for a general 3D tensor!"

# You can add more tests covering various dimensions and memory layouts if needed.
