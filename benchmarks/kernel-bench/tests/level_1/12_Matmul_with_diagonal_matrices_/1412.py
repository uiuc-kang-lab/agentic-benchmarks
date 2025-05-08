
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    return load(
        name="diag_matmul_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )

# Test case for issue 2: Passing tensors with non-float32 data type.
def test_non_float_dtype():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    # Create double precision tensors
    N = 1024
    M = 1024
    A = torch.randn(N, device="cuda", dtype=torch.float64)
    B = torch.randn(N, M, device="cuda", dtype=torch.float64)
    kernel_module = build_kernel()
    with pytest.raises(RuntimeError):
        # Expect an error or wrong behavior because the kernel is hard-coded for float
        C = kernel_module.forward(A, B)
        torch.cuda.synchronize()

# Test case for issue 3: Missing post-kernel error checking.
def test_invalid_dimensions_error():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    # Create tensors with mismatched dimensions. The forward function checks that A.size(0) == B.size(0).
    A = torch.randn(100, device="cuda", dtype=torch.float32)
    B = torch.randn(101, 50, device="cuda", dtype=torch.float32)
    kernel_module = build_kernel()
    with pytest.raises(RuntimeError):
        # The C++ forward should trigger a TORCH_CHECK error.
        C = kernel_module.forward(A, B)
        torch.cuda.synchronize()

# Test case for issue 1 (unused shared memory) and to ensure correctness for valid inputs.
def test_correct_result():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    # Compare result against PyTorch's native operation (torch.diag(A) @ B)
    N = 2048
    M = 512
    A = torch.randn(N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, M, device="cuda", dtype=torch.float32)
    
    kernel_module = build_kernel()
    # Call our CUDA kernel implementation
    C_kernel = kernel_module.forward(A, B)
    torch.cuda.synchronize()
    
    # Compute the reference result using PyTorch operations.
    C_ref = (torch.diag(A) @ B)
    
    assert torch.allclose(C_kernel, C_ref, atol=1e-5), \
        f"Kernel output does not match reference! Max difference: {(C_kernel - C_ref).abs().max()}"
