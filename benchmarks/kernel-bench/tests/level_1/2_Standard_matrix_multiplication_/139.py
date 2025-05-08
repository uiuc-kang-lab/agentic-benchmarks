
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the CUDA extension from kernel.cu
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

def test_issue_incorrect_thread_block_configuration():
    # This test uses small matrices to force the custom kernel path.
    # Due to the incorrect use of threadIdx.y, the custom kernel's computation is expected to be wrong.
    M, K, N = 64, 64, 64  # All dimensions below the 128 threshold.
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)
    my_module = build_kernel()
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    # Since threadIdx.y is never properly set, the kernel should produce a wrong result.
    assert not torch.allclose(C, C_ref, atol=1e-5), \
        "Kernel output unexpectedly matches reference output despite threadIdx.y misconfiguration."

def test_issue_shared_memory_reduction_error():
    # This test is designed to trigger the faulty shared memory reduction.
    # The reduction uses an incorrect iteration count that does not match WARPS_PER_BLOCK.
    M, K, N = 128, 32, 128  # A case where the threaded warps are in play.
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)
    my_module = build_kernel()
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    # We expect a discrepancy due to improper accumulation in shared memory.
    assert not torch.allclose(C, C_ref, atol=1e-5), \
        "Kernel reduction did not show the expected error from incorrect shared memory indexing."

def test_issue_warp_intrinsics_logic():
    # This test focuses on the misuse of warp shuffle intrinsics.
    # The b_reg handling with __shfl_up_sync is expected to result in an incorrect sum.
    M, K, N = 64, 64, 64  # Use small matrices to trigger custom kernel.
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)
    my_module = build_kernel()
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    # The error in b_reg propagation via shuffle is likely to yield an incorrect result.
    assert not torch.allclose(C, C_ref, atol=1e-5), \
        "Kernel output unexpectedly matches reference output despite suspected warp shuffle misuse."

def test_issue_non_multiple_of_warp_size():
    # This test provides a K dimension that is not a multiple of 32.
    # The kernel does not correctly account for remainder tiles, which should lead to an error.
    M, K, N = 64, 70, 64  # K=70 is not divisible by 32.
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)
    my_module = build_kernel()
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    assert not torch.allclose(C, C_ref, atol=1e-5), \
        "Kernel did not fail with non-divisible K dimension as expected."

def test_issue_cublas_memory_layout():
    # This test uses larger matrices to force the fallback to cuBLAS.
    # The issue here is that the argument order and assumed column-major layout may
    # lead to transposition errors relative to PyTorch's row-major layout.
    M, K, N = 256, 256, 256  # Dimensions beyond the custom kernel threshold.
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)
    my_module = build_kernel()
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    assert not torch.allclose(C, C_ref, atol=1e-5), \
        "cuBLAS fallback produced correct result, but an error due to memory layout was expected."
