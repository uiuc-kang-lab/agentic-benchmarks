
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    # Building the CUDA extension from kernel.cu
    module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

@pytest.fixture(scope="module")
def module():
    return build_kernel()

def test_non_contiguous(module):
    # Create a well‐shaped input but force a non‐contiguous layout via transpose.
    # The kernel expects contiguous memory.
    N, D = 128, 4096
    A = torch.randn(N, D, device="cuda", dtype=torch.float32).t()  # non-contiguous: shape (D, N)
    B = torch.randn(N, D, device="cuda", dtype=torch.float32).t()
    # Permute back to the intended shape if needed (simulate programmer mistake)
    with pytest.raises(RuntimeError, match="contiguous"):
        # When passing non‐contiguous tensors to data_ptr, the wrong layout will be used.
        # It is expected that if underlying assumptions are broken, a runtime error or
        # wrong result may occur. Here we assume the extension or the TORCH_CHECK
        # should catch the error.
        module.forward(A, B)

def test_wrong_dtype(module):
    # Passing tensors with wrong dtype (e.g., float64 instead of float32)
    N, D = 128, 4096
    A = torch.randn(N, D, device="cuda", dtype=torch.float64)
    B = torch.randn(N, D, device="cuda", dtype=torch.float64)
    with pytest.raises(RuntimeError, match="float32"):
        module.forward(A, B)

def test_wrong_dims(module):
    # Passing tensors that are not 2D.
    A = torch.randn(4096, device="cuda", dtype=torch.float32)  # 1D tensor
    B = torch.randn(4096, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="2D"):
        module.forward(A, B)

def test_mismatched_shapes(module):
    # Passing tensors with mismatched shapes.
    A = torch.randn(128, 4096, device="cuda", dtype=torch.float32)
    B = torch.randn(128, 2048, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="same shape"):
        module.forward(A, B)

def test_zero_dimension_vector(module):
    # Test the edge case where D==0.
    # Though unusual, if D==0, kernel should either do nothing or throw an error.
    N, D = 128, 0
    A = torch.empty(N, D, device="cuda", dtype=torch.float32)
    B = torch.empty(N, D, device="cuda", dtype=torch.float32)
    # Depending on design, the kernel might trigger a division by zero prevention
    # (eps is used in denominator but sqrtf(0) is 0) or may be considered invalid input.
    # Here, we expect the kernel to fail gracefully.
    with pytest.raises(RuntimeError):
        module.forward(A, B)
