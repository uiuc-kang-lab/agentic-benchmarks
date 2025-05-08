
import math
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="selu_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.fixture(scope="module")
def kernel_module():
    return build_kernel()

def test_cpu_input(kernel_module):
    # Issue 1: Kernel requires the input tensor to be on CUDA.
    x = torch.randn(10, 10, device="cpu", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="Input tensor must be a CUDA tensor"):
        kernel_module.forward(x)

def test_half_precision_input(kernel_module):
    # Issue 1: The kernel does not support half precision.
    x = torch.randn(16, 16, device="cuda", dtype=torch.half)
    with pytest.raises(RuntimeError):
        kernel_module.forward(x)
        
def test_non_contiguous_input(kernel_module):
    # Issue 2: The kernel assumes contiguous input.
    # Create a tensor and then make it non-contiguous by transposing.
    x = torch.randn(32, 64, device="cuda", dtype=torch.float32)
    x_noncontig = x.t()  # This transpose makes it non-contiguous.
    # Compute SELU using PyTorch for reference.
    y_ref = torch.selu(x_noncontig)
    # Compute SELU using our kernel.
    y_cuda = kernel_module.forward(x_noncontig)
    # The outputs might differ if kernel incorrectly assumes contiguity.
    # We expect the maximum absolute difference to be significant.
    diff = (y_ref - y_cuda).abs().max().item()
    assert diff > 1e-5, f"Expected a significant difference for non-contiguous input but got {diff}"

def test_3d_thread_block_indexing_limitation(kernel_module):
    # Issue 3: The kernel uses a 2D indexing scheme.
    # When a tensor requires a 3D grid for optimal mapping, the rigid 2D scheme might not cover the full range correctly.
    # We simulate this situation by providing a tensor with a large number of elements so that the computed grid dimensions would
    # ideally extend into 3D if generalized properly.
    numel = 2**18  # Large tensor.
    x = torch.randn(numel, device="cuda", dtype=torch.float32)
    y_ref = torch.selu(x)
    y_cuda = kernel_module.forward(x)
    # Even if the kernel processes all elements in a flat manner, its rigid indexing might lead to errors in a more general configuration.
    # Here, we check if an error is detectable (e.g., output difference) once the tensor size forces a high grid count.
    assert torch.allclose(y_ref, y_cuda, atol=1e-5) is False, "Expected the 2D indexing scheme to fail in a more general (3D) situation"
