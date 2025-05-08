
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

def softsign_cpu(x: torch.Tensor) -> torch.Tensor:
    return x / (1 + torch.abs(x))

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_input_tensor_type():
    # Issue 1: Passing a tensor with non-float32 dtype (e.g., float64) will trigger undefined behavior.
    # Expected behavior: The kernel should reject non-float32 types,
    # but since it does not, the output will be incorrect.
    x = torch.randn(1024, dtype=torch.float64, device="cuda")
    kernel_module = build_kernel()
    out = kernel_module.forward(x)
    
    # Compute a reference result in double precision.
    expected = softsign_cpu(x)
    # Because the kernel processes memory as if it were float32,
    # the result is very likely to differ from the expected double output.
    if torch.allclose(out.to(dtype=torch.float64), expected, atol=1e-5):
        pytest.fail("Kernel incorrectly processed a non-float32 tensor as if it were correct.")

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_memory_alignment():
    # Issue 2: The kernel assumes 16-byte alignment for vectorized accesses.
    # Create a sub-tensor (via slicing) that is contiguous but likely misaligned.
    x_full = torch.randn(1024 + 1, dtype=torch.float32, device="cuda")
    x = x_full[1:]  # This sub-tensor may not be 16-byte aligned.
    
    # Verify that x is contiguous but potentially misaligned.
    assert x.is_contiguous()
    
    kernel_module = build_kernel()
    out = kernel_module.forward(x)
    
    expected = softsign_cpu(x)
    # If the memory is misaligned, the vectorized loads/stores might return wrong results.
    if torch.allclose(out, expected, atol=1e-5):
        pytest.fail("Kernel produced correct output on a potentially misaligned tensor; expected misaligned access issues.")

