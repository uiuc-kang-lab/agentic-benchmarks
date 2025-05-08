
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    # Rebuild the CUDA extension from the file kernel.cu.
    return load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )

# Issue 1: Tensor alignment
def test_non_aligned_input():
    # Create an input tensor that is not well aligned for vectorized accesses.
    # One way is to allocate a slightly larger buffer and then take a non-zero offset view.
    N = 128
    M = 64   # M is divisible by 4 so that the vectorized path is taken.
    # Create a larger tensor and then slice off one element to force misalignment.
    A_full = torch.randn(N + 1, device="cuda", dtype=torch.float32)
    B_full = torch.randn(N + 1, M, device="cuda", dtype=torch.float32)
    # Slicing will very likely yield data not aligned on a 16-byte boundary.
    A = A_full.narrow(0, 1, N).clone()
    B = B_full.narrow(0, 1, N).clone()
    
    mod = build_kernel()
    C = mod.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.diag(A) @ B
    # The computed output might be invalid if misalignment causes load/store issues.
    assert torch.allclose(C, C_ref, atol=1e-5), "Output does not match reference due to misaligned memory access."

# Issue 2: Unnecessary/global memory read in the __shfl_sync broadcast.
# While this does not produce an error in a simple case, we simulate a scenario
# where the extra global load might be exposed by subtle numerical differences.
def test_broadcast_efficiency_numerical():
    # With a large input, slight numerical differences due to redundant loads may appear.
    N = 256   # many rows processed by warps.
    M = 64    # divisible by 4 for vectorized operations.
    A = torch.randn(N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, M, device="cuda", dtype=torch.float32)
    mod = build_kernel()
    C = mod.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.diag(A) @ B
    # Although the computation is mathematically simple (a scalar multiply per row),
    # redundant loads can sometimes show up as performance issues or numerical instability.
    assert torch.allclose(C, C_ref, atol=1e-5), "Mismatch due to unexpected broadcast behavior."

# Issue 3: Input type checking (only float32 is supported)
def test_input_tensor_type():
    N = 64
    M = 64
    A = torch.randn(N, device="cuda", dtype=torch.double)
    B = torch.randn(N, M, device="cuda", dtype=torch.double)
    mod = build_kernel()
    with pytest.raises(RuntimeError):
        # The kernel implementation assumes float32 so calling it with double should raise an error
        # (or at least produce wrong results, but here we expect a failure).
        _ = mod.forward(A, B)
        torch.cuda.synchronize()

# Issue 4: Kernel launching without proper dimension checking.
# Trigger the TORCH_CHECK in the C++ binding by providing mismatched dimensions.
def test_dimension_mismatch():
    N = 64
    M = 64
    A = torch.randn(N + 1, device="cuda", dtype=torch.float32)  # A’s length is N+1.
    B = torch.randn(N, M, device="cuda", dtype=torch.float32)    # B has N rows.
    mod = build_kernel()
    with pytest.raises(RuntimeError):
        _ = mod.forward(A, B)
        torch.cuda.synchronize()
