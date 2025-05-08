
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    module = load(
        name="test_softsign",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

def run_kernel_forward(x: torch.Tensor):
    cuda_module = build_kernel()
    # Call the forward CUDA kernel function
    out = cuda_module.forward(x)
    torch.cuda.synchronize()
    return out

# Test case for Issue 1: passing a non-float32 (e.g., double) tensor
def test_wrong_dtype():
    x = torch.randn(16, 16384, dtype=torch.double, device='cuda')
    with pytest.raises(RuntimeError):
        # The kernel expects float32 data but we pass double; error should occur.
        run_kernel_forward(x)

# Test case for Issue 2: missing error checking leads to silent failure when a non-contiguous tensor is given.
def test_non_contiguous():
    # Create a contiguous tensor and then make it non-contiguous via transpose
    x = torch.randn(16, 16384, device='cuda', dtype=torch.float32)
    x = x.t()  # this makes the tensor non-contiguous
    with pytest.raises(RuntimeError):
        run_kernel_forward(x)

# Test case for Issue 3: using a CPU tensor (violates the CUDA check) should trigger an error.
def test_cpu_tensor():
    x = torch.randn(16, 16384, device='cpu', dtype=torch.float32)
    with pytest.raises(RuntimeError):
        run_kernel_forward(x)
