
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

# Issue 1: The kernel only supports float32 inputs.
def test_input_tensor_type():
    my_module = build_kernel()
    M, K, N = 32, 16, 24
    # Create tensors with double type instead of float32.
    A = torch.randn(M, K, dtype=torch.float64, device='cuda')
    B = torch.randn(K, N, dtype=torch.float64, device='cuda')
    with pytest.raises(RuntimeError):
        # The kernel does not check the type, so it will misinterpret the data and eventually lead to an error.
        _ = my_module.forward(A, B)
    torch.cuda.synchronize()

# Issue 2: The kernel assumes contiguous inputs.
def test_non_contiguous_inputs():
    my_module = build_kernel()
    M, K, N = 32, 16, 24
    # Create contiguous tensors and then make them non-contiguous by transposing.
    A = torch.randn(K, M, device='cuda', dtype=torch.float32).t()  # t() makes it non-contiguous.
    B = torch.randn(N, K, device='cuda', dtype=torch.float32).t()  # non-contiguous.
    with pytest.raises(RuntimeError):
        # The CHECK_INPUT macro should trigger on non-contiguous tensors.
        _ = my_module.forward(A, B)
    torch.cuda.synchronize()

# Issue 3: Lack of robust CUDA error checking after kernel launch.
def test_invalid_dimensions():
    my_module = build_kernel()
    # Deliberately mismatch dimensions to trigger a check and observe error propagation.
    M, K, N = 32, 16, 24
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    # B has wrong first dimension.
    B = torch.randn(K + 1, N, device='cuda', dtype=torch.float32)
    with pytest.raises(RuntimeError):
        # The TORCH_CHECK in the host function should reject input tensors with mismatched dimensions.
        _ = my_module.forward(A, B)
    torch.cuda.synchronize()
