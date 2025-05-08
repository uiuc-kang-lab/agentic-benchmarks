
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="softsign_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

def softsign_ref(x: torch.Tensor) -> torch.Tensor:
    return x / (1.0 + torch.abs(x))

def test_non_float_input():
    """
    Issue 1:
    This test passes a double-precision tensor to the kernel.
    Since the kernel blindly casts the input pointer to float*, the output will be computed incorrectly.
    We check that the kernel output does not match the correct softsign computed in double precision.
    """
    module = build_kernel()
    # Create a double precision tensor on CUDA.
    x = torch.randn(1024, dtype=torch.double, device="cuda")
    # Invoke the CUDA kernel (which expects float) with a double tensor.
    out = module.forward(x)
    # Compute reference output using native PyTorch (with proper conversion)
    ref = softsign_ref(x)
    # The outputs are expected NOT to be allclose because the kernel misinterprets the input data.
    assert not torch.allclose(out.double(), ref, atol=1e-5), "Kernel incorrectly handled double precision input!"

def test_non_contiguous():
    """
    Issue 2:
    This test feeds a non-contiguous tensor to the CUDA kernel.
    The CHECK_CONTIGUOUS macro in the kernel should trigger a runtime error.
    """
    module = build_kernel()
    # Create a contiguous tensor and then make it non-contiguous by transposing.
    x = torch.randn(32, 32, device="cuda", dtype=torch.float32).t()  # non-contiguous tensor
    with pytest.raises(RuntimeError, match="must be contiguous"):
        module.forward(x)
