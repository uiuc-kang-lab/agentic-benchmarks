
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

# Issue 1: Kernel only supports float32.
def test_non_float_dtype():
    # Use double precision inputs.
    M, K, N = 128, 64, 128
    # Create inputs on CUDA with dtype float64.
    A = torch.randn(M, K, dtype=torch.float64, device="cuda")
    B = torch.randn(K, N, dtype=torch.float64, device="cuda")
    kernel_module = build_kernel()
    # The kernel expects float32 and will treat the inputs as float32,
    # so the result will be numerically incorrect compared to the reference.
    C = kernel_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    # We expect a significant difference.
    assert not torch.allclose(C, C_ref, atol=1e-3), (
        "Kernel unexpectedly produced correct output for float64 inputs."
    )

# Issue 2: Missing proper CUDA error checking after kernel launch.
def test_cpu_input_error():
    # Pass CPU tensors to trigger the host check that inputs must be CUDA.
    M, K, N = 64, 32, 64
    A = torch.randn(M, K, device="cpu", dtype=torch.float32)
    B = torch.randn(K, N, device="cpu", dtype=torch.float32)
    kernel_module = build_kernel()
    with pytest.raises(Exception) as excinfo:
        _ = kernel_module.forward(A, B)
    assert "Input tensors must be on CUDA devices" in str(excinfo.value), (
        "Expected error for CPU input tensors was not raised."
    )

# Issue 3: Shared memory declarations inside the loop.
def test_shared_memory_usage():
    # We choose dimensions that force multiple iterations over the K-dimension.
    # Even though the atomicAdd branch is not triggered (gridDim.z == 1 if possible)
    # we use a K that is many times TILE_K to force multiple iterations.
    M, tile = 64, 16
    # Set K such that there are multiple k-tiles in a loop.
    K = tile * 5 + 3  # 5 full tiles plus an extra partial tile
    N = 64
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    kernel_module = build_kernel()
    C = kernel_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    # The kernel may be correct (so the result matches) but the inefficiency or structure
    # (shared memory inside loop) is a design issue.
    assert torch.allclose(C, C_ref, atol=1e-4), (
        f"Kernel output differs from torch.matmul for multiple k-tiles. Max diff: {(C - C_ref).abs().max()}"
    )

# Issue 4: Grid partitioning along the K-dimension and atomicAdd path.
def test_k_dimension_partitioning():
    # Select sizes that force gridDim.z > 1 so that multiple thread blocks accumulate
    # partial results using atomicAdd.
    # Here, choose K big enough so that (K + TILE_K - 1) / TILE_K > 1.
    M, N = 256, 256
    K = 64 * 4  # Many k-tiles
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    kernel_module = build_kernel()
    # When gridDim.z > 1, the kernel initializes C to zeros and uses atomicAdd.
    C = kernel_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    # Even if the kernel is correct, the use of atomicAdd may cause heavy contention
    # in more complex scenarios. For this simple test we at least require correct accumulation.
    assert torch.allclose(C, C_ref, atol=1e-4), (
        f"Kernel output with atomic accumulation differs from torch.matmul. Max diff: {(C - C_ref).abs().max()}"
    )
