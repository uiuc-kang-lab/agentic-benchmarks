
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    return load(
        name="triangular_mm_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

# Helper function: lower-triangular multiplication using PyTorch's matmul and tril.
def ref_triangular_mm(A, B):
    return torch.tril(torch.matmul(A, B))

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_incorrect_tile_range():
    # Test case to reveal an incorrect summation loop iteration range.
    # Choose a size large enough that the incorrect k_tile iteration (driven by blockIdx.x and blockIdx.y)
    # produces a noticeable deviation from the reference lower-triangular matmul output.
    N = 128  # Use a moderate size where tile indices matter.
    # Create lower triangular matrices.
    A = torch.tril(torch.randn(N, N, device="cuda", dtype=torch.float32))
    B = torch.tril(torch.randn(N, N, device="cuda", dtype=torch.float32))
    
    mod = build_kernel()
    C_kernel = mod.forward(A, B)
    torch.cuda.synchronize()
    
    C_ref = ref_triangular_mm(A, B)
    # With the incorrect k_tile loop, the kernel computation is likely to produce wrong values.
    # We check that the maximum absolute difference exceeds a small tolerance.
    max_diff = (C_ref - C_kernel).abs().max().item()
    assert max_diff > 1e-3, f"Kernel appears correct but was expected to be wrong (max difference {max_diff})"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_boundary_check():
    # Test case to trigger potential boundary issues when N is not a multiple of TILE_SIZE.
    # Here, N is deliberately chosen to be not divisible by TILE_SIZE.
    N = 70  # 70 is not a multiple of TILE_SIZE (32)
    A = torch.tril(torch.randn(N, N, device="cuda", dtype=torch.float32))
    B = torch.tril(torch.randn(N, N, device="cuda", dtype=torch.float32))
    
    mod = build_kernel()
    try:
        C_kernel = mod.forward(A, B)
        torch.cuda.synchronize()
    except Exception as e:
        pytest.fail(f"Kernel raised an exception for non-multiple-of-tile dimension: {e}")
    
    C_ref = ref_triangular_mm(A, B)
    # With the missing boundary check in the inner loop, the output may differ from expected.
    max_diff = (C_ref - C_kernel).abs().max().item()
    assert max_diff > 1e-3, f"Kernel output almost matches reference despite expected boundary issues (max diff {max_diff})"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_input_tensor_type():
    # Test case to check that the kernel behaves badly or produces wrong output when using a type other than float32.
    N = 64
    # Create lower triangular matrices with double precision.
    A = torch.tril(torch.randn(N, N, device="cuda", dtype=torch.float64))
    B = torch.tril(torch.randn(N, N, device="cuda", dtype=torch.float64))
    
    mod = build_kernel()
    # The kernel is hardcoded for float so the double data may be reinterpreted,
    # leading to incorrect results.
    C_kernel = mod.forward(A, B)
    torch.cuda.synchronize()
    
    C_ref = ref_triangular_mm(A.float(), B.float())
    max_diff = (C_ref - C_kernel).abs().max().item()
    assert max_diff > 1e-3, f"Kernel output for double precision input unexpectedly matches the reference (max diff {max_diff})"
