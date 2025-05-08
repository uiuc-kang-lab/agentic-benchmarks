
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    # Load the CUDA kernel extension.
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_input_tensor_type():
    # Issue 5: Kernel does not support non-float32 types
    N = 128
    A = torch.randn(N, N, dtype=torch.double, device='cuda')
    B = torch.randn(N, N, dtype=torch.double, device='cuda')
    module = build_kernel()
    with pytest.raises(RuntimeError):
        _ = module.forward(A, B)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tile_boundary_conditions():
    # Issue 3 & 4: Use a size not divisible by TILE_SIZE (32) or VECTOR_SIZE (4)
    # In such cases, the vectorized loads may cause out-of-bound accesses or race conditions,
    # and the result will differ from torch.matmul.
    N = 70  # not divisible by 32 nor by 4 in a uniform way.
    A = torch.randn(N, N, dtype=torch.float32, device="cuda")
    B = torch.randn(N, N, dtype=torch.float32, device="cuda")
    module = build_kernel()
    C = module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    # We expect a mismatch due to race conditions or out-of-bound accesses.
    assert not torch.allclose(C, C_ref, atol=1e-5), \
        f"Kernel output unexpectedly matches reference on non-divisible dimensions."

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_vectorized_load_issue():
    # Issue 1 & 2: Force conditions where the vectorized load branch is taken.
    # We build a matrix where the indexing will often satisfy the (col % 4 == 0) and ((m*TILE_SIZE+tx)%4==0) conditions.
    # The overlap in writes should cause incorrect results.
    N = 128  # multiple of TILE_SIZE = 32 but still can trigger vectorized load branch.
    # Create matrices whose dimensions force the vectorized loading conditions in many threads.
    # Using sequential values to magnify potential race conditions.
    A = torch.arange(N*N, dtype=torch.float32, device="cuda").view(N, N) * 1e-3
    B = torch.arange(N*N, dtype=torch.float32, device="cuda").view(N, N) * 1e-3
    module = build_kernel()
    C = module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    # The incorrect overlapping writes should lead to a difference.
    assert not torch.allclose(C, C_ref, atol=1e-5), \
        f"Kernel output unexpectedly matches reference, indicating potential vectorized load issues."

