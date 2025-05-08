
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the CUDA extension from kernel.cu.
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test case 1: Pass input tensors with type other than float32 (e.g. double) to trigger type-related issues.
def test_input_tensor_wrong_dtype():
    M, N, K = 128, 128, 32
    A = torch.randn(M, K, dtype=torch.double, device='cuda')  # wrong dtype
    B = torch.randn(K, N, dtype=torch.double, device='cuda')  # wrong dtype
    kernel_module = build_kernel()
    with pytest.raises(RuntimeError):
        # Expect a runtime error since the tensors are not float32, but the kernel uses data_ptr<float>()
        C = kernel_module.forward(A, B)
        torch.cuda.synchronize()

# Test case 2: Pass input tensors with mismatched inner dimensions to trigger shape checking issues.
def test_input_tensor_shape_mismatch():
    M, N, K = 128, 128, 32
    # Create A with shape (M, K) and B with mismatched shape, e.g. (K+1, N)
    A = torch.randn(M, K, dtype=torch.float32, device='cuda')
    B = torch.randn(K+1, N, dtype=torch.float32, device='cuda')
    kernel_module = build_kernel()
    with pytest.raises(RuntimeError):
        # The kernel does not check that A and B are conformable.
        # The out-of-bound access should eventually trigger an error upon synchronization.
        C = kernel_module.forward(A, B)
        torch.cuda.synchronize()
