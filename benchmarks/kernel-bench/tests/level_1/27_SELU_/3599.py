
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper function to build the extension module from kernel.cu
def build_kernel():
    cuda_module = load(
        name="selu_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Trigger issue with unsupported type (half precision)
def test_half_precision_not_supported():
    kernel_module = build_kernel()
    # Create a half precision tensor on CUDA
    x = torch.randn(128, device="cuda").half()
    with pytest.raises(RuntimeError):
        # This should raise an error because our dispatch does not support half
        _ = kernel_module.forward(x)

# Test 2: Trigger issue with non-contiguous tensor input
def test_non_contiguous_input():
    kernel_module = build_kernel()
    # Create a contiguous tensor and then take a non-contiguous slice
    x = torch.randn(32, 64, device="cuda")
    non_contig = x.t()  # transposed, likely non-contiguous
    # We compute reference using torch.selu (which handles non-contiguity correctly)
    ref = torch.selu(non_contig)
    out = kernel_module.forward(non_contig)
    # The two may not match if the kernel assumed contiguous memory
    # Note: we use allclose with a relatively loose tolerance since there might be minor differences.
    assert not torch.allclose(out, ref, atol=1e-5), "Kernel unexpectedly worked with non-contiguous input."

# Test 3: Trigger issue with missing kernel launch error checking by intentionally sending a CPU tensor
def test_cpu_input_error():
    kernel_module = build_kernel()
    # Create CPU tensor; our kernel requires a CUDA tensor.
    x_cpu = torch.randn(128)
    with pytest.raises(RuntimeError):
        _ = kernel_module.forward(x_cpu)
