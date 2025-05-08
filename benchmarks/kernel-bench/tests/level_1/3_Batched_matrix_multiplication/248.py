
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_dtype_support():
    # Issue 1: Kernel only supports float32.
    # Create double precision inputs.
    batch_size, M, K, N = 2, 64, 96, 128
    A = torch.randn(batch_size, M, K, device="cuda", dtype=torch.float64)
    B = torch.randn(batch_size, K, N, device="cuda", dtype=torch.float64)
    module = build_kernel()
    # The kernel will treat the underlying bits as float32.
    C = module.forward(A, B)
    torch.cuda.synchronize()
    # Compute reference using bmm casted to float32 manually (simulate what would happen)
    A_f32 = A.to(torch.float32)
    B_f32 = B.to(torch.float32)
    C_ref = torch.bmm(A_f32, B_f32)
    # Since the kernel misinterprets double data as float,
    # the results should NOT match the reference.
    assert not torch.allclose(C, C_ref, atol=1e-4), (
        "Kernel unexpectedly handled non-float32 tensors correctly."
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_non_contiguous_inputs():
    # Issue 2: Kernel assumes contiguous memory.
    batch_size, M, K, N = 2, 64, 96, 128
    A = torch.randn(batch_size, M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(batch_size, K, N, device="cuda", dtype=torch.float32)
    
    # Make A and B non-contiguous by a transpose operation (but still 3D)
    # For example, swap the last two dimensions and then swap back later.
    A_noncontig = A.transpose(1, 2)
    B_noncontig = B.transpose(1, 2)
    
    # Though logically the same, now they are non-contiguous.
    module = build_kernel()
    # The kernel uses .data_ptr<float>() assuming standard row-major contiguous layout.
    C = module.forward(A_noncontig, B_noncontig)
    torch.cuda.synchronize()
    
    # Compute reference using torch.bmm.
    C_ref = torch.bmm(A_noncontig, B_noncontig)
    
    # Due to incorrect memory layout interpretation, the results should be different.
    assert not torch.allclose(C, C_ref, atol=1e-4), (
        "Kernel unexpectedly handled non-contiguous inputs correctly."
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_non_divisible_dimensions():
    # Issue 3: Fixed TILE_SIZE can cause warp divergence inefficiencies.
    # Here we choose dimensions that are not multiples of 32.
    batch_size = 3
    M = 45   # Not divisible by 32
    K = 53   # Not divisible by 32
    N = 61   # Not divisible by 32
    A = torch.randn(batch_size, M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(batch_size, K, N, device="cuda", dtype=torch.float32)
    
    module = build_kernel()
    C = module.forward(A, B)
    torch.cuda.synchronize()
    
    # Using torch.bmm as a reference (it handles this case correctly)
    C_ref = torch.bmm(A, B)
    
    # Even though the kernel guards against out-of-bound accesses,
    # its hardcoded TILE_SIZE might lead to subtle errors.
    # We expect a discrepancy if the edge-case handling is suboptimal.
    assert not torch.allclose(C, C_ref, atol=1e-4), (
        "Kernel output nearly matches torch.bmm even for non-multiple dimensions; "
        "this suggests that corner cases were not triggered."
    )
