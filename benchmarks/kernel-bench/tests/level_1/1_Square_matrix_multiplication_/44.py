
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Kernel only supports square matrices.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_non_square_matrix():
    module = build_kernel()
    # Create non-square matrices (e.g., (32, 64) and (64, 32)) which should trigger the square-check.
    A = torch.randn(32, 64, dtype=torch.float32, device="cuda")
    B = torch.randn(64, 32, dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError) as excinfo:
        module.forward(A, B)
    assert "square" in str(excinfo.value).lower()

# Issue 2: Kernel only supports float32 tensors.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_non_float32_input():
    module = build_kernel()
    N = 64
    A = torch.randn(N, N, dtype=torch.float64, device="cuda")  # double precision instead of float32
    B = torch.randn(N, N, dtype=torch.float64, device="cuda")
    with pytest.raises(RuntimeError) as excinfo:
        module.forward(A, B)
    assert "float32" in str(excinfo.value)

# Issue 3: Kernel requires contiguous input tensors.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_non_contiguous_tensor():
    module = build_kernel()
    N = 64
    # Create a non-contiguous tensor by transposing an otherwise contiguous tensor.
    A = torch.randn(N, N, dtype=torch.float32, device="cuda").t()  # non-contiguous due to transpose
    B = torch.randn(N, N, dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError) as excinfo:
        module.forward(A, B)
    assert "contiguous" in str(excinfo.value).lower()

# Issue 4: Kernel not supporting batched operations (i.e. inputs with more than 2 dimensions).
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_batched_input():
    module = build_kernel()
    batch = 4
    N = 64
    # Create batched matrices (3D tensor) which the kernel is not designed for.
    A = torch.randn(batch, N, N, dtype=torch.float32, device="cuda")
    B = torch.randn(batch, N, N, dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError):
        module.forward(A, B)
