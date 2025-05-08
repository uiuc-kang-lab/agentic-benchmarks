
import torch
import pytest
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

# Test 1: Trigger issue with non-float32 tensors. The kernel only supports float and should fail or produce wrong result.
def test_non_float32_dtype():
    cuda_module = build_kernel()
    N = 64
    # Create double tensors which are not supported by our kernel that expects float32
    A = torch.randn(N, N, dtype=torch.double, device='cuda')
    B = torch.randn(N, N, dtype=torch.double, device='cuda')
    with pytest.raises(RuntimeError):
        # We expect that if the wrong dtype is passed, conversion error or invalid memory access may occur.
        C = cuda_module.forward(A, B)
        torch.cuda.synchronize()

# Test 2: Trigger issue with non-contiguous input tensors.
def test_non_contiguous_inputs():
    cuda_module = build_kernel()
    N = 64
    # Create contiguous tensors then make them non-contiguous by transposing
    A = torch.randn(N, N, dtype=torch.float32, device='cuda').t()  # non-contiguous view
    B = torch.randn(N, N, dtype=torch.float32, device='cuda').t()  # non-contiguous view
    # Even though our kernel mechanically supports 2D access via strides,
    # our implementation does not explicitly check contiguity and may yield wrong results.
    C = cuda_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    # This test is designed to expose potential issues if non-contiguous memory layout causes wrong reads.
    assert not torch.allclose(C, C_ref, atol=1e-5), "Kernel should miscompute with non-contiguous inputs, but results match!"

# Test 3: Error checking after kernel launch.
def test_kernel_error_checking():
    cuda_module = build_kernel()
    # Intentionally pass in tensors that lead to dimension mismatch to trigger an error
    A = torch.randn(32, 16, dtype=torch.float32, device='cuda')
    B = torch.randn(17, 32, dtype=torch.float32, device='cuda')  # Wrong inner dimension
    with pytest.raises(Exception):
        C = cuda_module.forward(A, B)
        torch.cuda.synchronize()

# Test 4: Handling only 2D inputs (lack of support for batched matrix multiplication).
def test_batched_input():
    cuda_module = build_kernel()
    # Create 3D tensor inputs to simulate batched matrix multiplication.
    A = torch.randn(10, 32, 16, dtype=torch.float32, device='cuda')
    B = torch.randn(10, 16, 32, dtype=torch.float32, device='cuda')
    # Our launcher checks A.dim() != 2 and should throw an exception.
    with pytest.raises(RuntimeError):
        C = cuda_module.forward(A, B)
        torch.cuda.synchronize()
