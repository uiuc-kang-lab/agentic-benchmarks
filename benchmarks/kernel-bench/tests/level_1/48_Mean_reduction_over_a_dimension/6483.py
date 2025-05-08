
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the CUDA extension module from kernel.cu
    cuda_module = load(
        name="kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_shared_memory_double():
    """
    This test uses a double tensor. Due to issue 1 (fixed shared memory size),
    the kernel will allocate shared memory based on sizeof(float) rather than double.
    This should trigger incorrect results when the input tensor is double.
    """
    mod = build_kernel()
    batch_size, dim1, dim2 = 16, 128, 128
    # Create an input tensor with double precision.
    x = torch.randn(batch_size, dim1, dim2, dtype=torch.double, device='cuda')
    # Use reduction along dimension 1.
    out = mod.forward(x, 1)
    expected = torch.mean(x, dim=1)
    # If the kernel were correct, these would be close.
    # We expect them not to be close due to the shared memory misallocation.
    if torch.allclose(out, expected, atol=1e-5):
        raise AssertionError("Kernel produced correct results for double tensor, "
                             "but it should fail due to shared memory size issue.")

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_non_contiguous():
    """
    This test creates a non-contiguous tensor by transposing a contiguous tensor.
    Because the kernel assumes a contiguous input with a specific memory layout,
    it is expected to produce incorrect results when handling a non-contiguous tensor.
    """
    mod = build_kernel()
    # Create a contiguous tensor and then transpose to make it non-contiguous.
    x = torch.randn(32, 64, 128, device='cuda', dtype=torch.float32)
    x_noncontig = x.transpose(0, 1)  # This makes the tensor non-contiguous.
    # Reduce along dimension 1 (the one that is affected by the transpose).
    out = mod.forward(x_noncontig, 1)
    expected = torch.mean(x_noncontig, dim=1)
    if torch.allclose(out, expected, atol=1e-5):
        raise AssertionError("Kernel produced correct results for a non-contiguous tensor, "
                             "but it assumes a contiguous layout and should misbehave in general cases.")
