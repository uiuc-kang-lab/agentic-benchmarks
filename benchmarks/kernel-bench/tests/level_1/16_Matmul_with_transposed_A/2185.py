
import pytest
import time
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

# Issue 1: Non‐contiguous input tensor.
# We manufacture a non‐contiguous view of A that has the same size but wrong strides.
def test_non_contiguous_input():
    M, K, N = 1024, 4096, 2048
    # Create contiguous inputs in the expected shape (K, M) and (K, N)
    A = torch.randn(K, M, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)
    # Create a non-contiguous view by “corrupting” the stride of A.
    # (This view has the same size but wrong memory layout so that the kernel’s indexing is off.)
    A_noncontig = torch.as_strided(A, size=A.size(), stride=(A.stride(0), A.stride(1) + 1))
    assert not A_noncontig.is_contiguous(), "Expected A_noncontig to be non contiguous."

    my_module = build_kernel()
    C = my_module.forward(A_noncontig, B)
    # Compute the reference result using PyTorch operations (which handle non contiguity correctly)
    C_ref = torch.matmul(A_noncontig.T, B)
    # The results will differ because the kernel assumed contiguous (stride == [M,1])
    assert not torch.allclose(C, C_ref, atol=1e-5), "Kernel computed a result as if A were contiguous!"

# Issue 2: Grid dimension z exceeding device limits.
def test_large_K_dimension():
    # With BLOCK_K defined as 32 in the kernel, gridDim.z = ceil(K / 32) must be <= 65535.
    # Here we choose K so that grid.z becomes 65536 (or higher) to trigger a launch error.
    BLOCK_K = 32
    max_grid_z = 65535
    K = (max_grid_z + 1) * BLOCK_K  # This K makes grid.z > 65535.
    # Keep M and N small to not use excessive memory.
    M, N = 16, 16
    A = torch.randn(K, M, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)
    my_module = build_kernel()
    with pytest.raises(RuntimeError):
        # The launch should fail because gridDim.z exceeds the maximum allowed.
        C = my_module.forward(A, B)
        torch.cuda.synchronize()

# Issue 3: Input tensor type is not float32.
def test_wrong_dtype():
    M, K, N = 1024, 4096, 2048
    A = torch.randn(K, M, device='cuda', dtype=torch.float64)
    B = torch.randn(K, N, device='cuda', dtype=torch.float64)
    my_module = build_kernel()
    with pytest.raises(RuntimeError):
        C = my_module.forward(A, B)
        torch.cuda.synchronize()

# Issue 4: Atomic add contention leads to potential performance degradation.
# Although this does not cause inaccuracy, the heavy use of atomicAdd may slow down the kernel.
# This test benchmarks a moderately high-K case and asserts that the execution time is under an arbitrary threshold.
def test_atomic_add_performance():
    M, K, N = 256, 8192, 256  # Increase K to have many atomic adds.
    A = torch.randn(K, M, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)
    my_module = build_kernel()
    start = time.time()
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    end = time.time()
    exec_time = end - start
    # The threshold is arbitrarily chosen; if the atomic contention is severe, the runtime will be high.
    assert exec_time < 0.5, f"Kernel execution time is too high ({exec_time:.4f} s), which may be due to atomicAdd contention."

# Issue 5: Insufficient synchronization for error reporting.
# We induce an error by providing mismatched dimensions so that the atomicAdd writes out-of-bounds.
def test_error_synchronization():
    M, K, N = 1024, 4096, 2048
    # Create A with an extra row (mismatching the expected K dimension)
    A = torch.randn(K + 1, M, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)
    my_module = build_kernel()
    with pytest.raises(RuntimeError):
        C = my_module.forward(A, B)
        # Force synchronization so that any asynchronous kernel error is raised.
        torch.cuda.synchronize()
