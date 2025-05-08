
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

# Issue 1: Data type limitation (only float32 supported)
def test_input_tensor_type():
    # Create double precision (float64) inputs
    N = 128
    A = torch.tril(torch.randn(N, N, device="cuda", dtype=torch.float64))
    B = torch.tril(torch.randn(N, N, device="cuda", dtype=torch.float64))
    my_module = build_kernel()
    # Expect the kernel to fail check or produce incorrect results 
    with pytest.raises(RuntimeError):
        C = my_module.forward(A, B)
        torch.cuda.synchronize()

# Issue 2: Lack of batched input support (only 2D tensors accepted)
def test_batched_input():
    N = 64
    batch = 4
    # Create a batched lower triangular matrix input (incorrect shape)
    A = torch.tril(torch.randn(batch, N, N, device="cuda", dtype=torch.float32))
    B = torch.tril(torch.randn(batch, N, N, device="cuda", dtype=torch.float32))
    my_module = build_kernel()
    # The kernel expects 2D tensors, so this should trigger an error
    with pytest.raises(RuntimeError):
        C = my_module.forward(A, B)
        torch.cuda.synchronize()

# Issue 3: Suboptimal thread utilization and potential memory access inefficiency
def test_non_divisible_dimensions():
    # Use a matrix size that is not a multiple of BLOCK_SIZE (32)
    N = 70  # 70 is not divisible by 32, which can expose issues with the bounds logic and load patterns
    A = torch.tril(torch.randn(N, N, device="cuda", dtype=torch.float32))
    B = torch.tril(torch.randn(N, N, device="cuda", dtype=torch.float32))
    my_module = build_kernel()
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    # The reference multiplication must also restrict to lower triangular result
    C_ref_full = torch.matmul(A, B)
    C_ref = torch.tril(C_ref_full)
    # Because of potential inefficiencies or boundary condition issues,
    # the results might be slightly off. Use a tight tolerance.
    assert torch.allclose(C, C_ref, atol=1e-5), f"Kernel output differs from reference output! Max difference: {(C - C_ref).abs().max()}"
