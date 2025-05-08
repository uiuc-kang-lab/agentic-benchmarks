
import torch
import pytest
from torch.utils.cpp_extension import load
import math

# Build and load the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="gelu_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# A reference implementation of GELU matching the kernel's formula (using tanhf)
def gelu_reference(x: torch.Tensor) -> torch.Tensor:
    # Use the same constants as in the kernel
    sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
    coeff = 0.044715
    return 0.5 * x * (1.0 + torch.tanh(sqrt_2_over_pi * (x + coeff * x.pow(3))))

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tail_elements():
    """
    Test case for Issue 1: Tail elements (when total elements isn't divisible by 4)
    The kernel loads zeros for the tail elements instead of the correct values.
    """
    # Create an input tensor with a total number of elements that is not a multiple of 4.
    # Here, we use a 2D tensor shape which overall produces a number of elements not divisible by 4.
    batch_size, dim = 17, 13  # 17*13 = 221, which is not divisible by 4
    x = torch.randn(batch_size, dim, dtype=torch.float32, device='cuda')
    
    # Get the expected result using the reference GELU computation.
    y_expected = gelu_reference(x)
    
    # Run the custom CUDA kernel.
    gelu_cuda = build_kernel()
    y_cuda = gelu_cuda.forward(x)
    
    torch.cuda.synchronize()
    
    # The kernel incorrectly handles the tail elements.
    # We expect a noticeable difference in at least one element.
    if torch.allclose(y_expected, y_cuda, atol=1e-5):
        raise AssertionError("Tail elements not handled correctly: CUDA kernel output matches reference.")
    
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_input_alignment():
    """
    Test case for Issue 2: Misaligned input tensor.
    The kernel assumes proper 16-byte alignment for vectorized accesses.
    We create a misaligned tensor by slicing off a single element.
    """
    # Create a 1D tensor with enough elements.
    N = 1024 + 1  # extra element ensures that the slice [1:] is misaligned
    base = torch.randn(N, dtype=torch.float32, device='cuda')
    
    # Slicing [1:] produces a tensor that is contiguous but with a storage offset,
    # and therefore its address may not be 16-byte aligned.
    x = base[1:]
    assert x.is_contiguous(), "Sliced tensor must be contiguous"
    
    # Expected output computed using the reference GELU
    y_expected = gelu_reference(x)
    
    # Run the custom CUDA kernel.
    gelu_cuda = build_kernel()
    y_cuda = gelu_cuda.forward(x)
    
    torch.cuda.synchronize()
    
    # Due to alignment issues, the kernel may produce results that do not match the reference.
    if torch.allclose(y_expected, y_cuda, atol=1e-5):
        raise AssertionError("Alignment issue not triggered: CUDA kernel output matches reference despite misaligned input.")
