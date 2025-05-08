
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the module from the "kernel.cu" file.
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1: Kernel only supports float32; using double should trigger an error or wrong result.
def test_input_tensor_type():
    my_module = build_kernel()
    N = 32
    # Create double tensors on CUDA
    A = torch.randn(N, N, dtype=torch.double, device="cuda")
    B = torch.randn(N, N, dtype=torch.double, device="cuda")
    with pytest.raises(RuntimeError):
        # The kernel uses A.data_ptr<float>(), so passing doubles should trigger an error.
        C = my_module.forward(A, B)
    torch.cuda.synchronize()

# Issue 2: Kernel does not support batched inputs.
def test_batched_input():
    my_module = build_kernel()
    batch = 4
    M, K, N = 32, 48, 40
    # Create batched matrices (3D tensors)
    A = torch.randn(batch, M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(batch, K, N, device="cuda", dtype=torch.float32)
    with pytest.raises(IndexError):
        # The kernel expects 2D matrices, so passing 3D tensors should cause an error.
        C = my_module.forward(A, B)
    torch.cuda.synchronize()

# Issue 3: Kernel does not verify inner dimensions compatibility.
def test_incompatible_inner_dims():
    my_module = build_kernel()
    M, K1, K2, N = 32, 48, 50, 40  # K1 != K2
    A = torch.randn(M, K1, device="cuda", dtype=torch.float32)
    B = torch.randn(K2, N, device="cuda", dtype=torch.float32)
    # Depending on how the kernel is used, this may not trigger an explicit error but will produce wrong results.
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B[:K1, :])  # Force a reference computation with mismatched dims
    # The outputs should not match because of the inner dimension mismatch.
    assert not torch.allclose(C, C_ref, atol=1e-5), "Kernel incorrectly accepted mismatched inner dimensions."

# Issue 4: Fixed tile sizes may perform poorly or incorrectly for irregular shapes.
def test_irregular_tile_sizes():
    my_module = build_kernel()
    # Choose dimensions that are not multiples of TILE_DIM (which is BLOCK_SIZE*2 = 32).
    M, K, N = 37, 45, 53
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    # In an ideal implementation, the result should be close; here we check that differences appear.
    # If the tiling did not correctly handle the edge cases, then the maximum error will be significant.
    diff = (C - C_ref).abs().max().item()
    assert diff > 1e-3, f"Kernel output unexpectedly matches reference for irregular shapes (max diff: {diff})."

# Issue 5: Shared memory bank conflict potential.
# Although bank conflicts are a performance issue rather than a correctness bug,
# we can at least check that for certain matrix sizes the kernel output is not optimal.
def test_shared_memory_bank_conflicts():
    my_module = build_kernel()
    # Use a moderately sized matrix to see if performance suffers or results are degraded.
    M, K, N = 512, 512, 512
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    # Run many iterations to expose performance degradation.
    for _ in range(10):
        C = my_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    # We are testing correctness here, but note that potential bank conflicts would reduce performance.
    # Thus, we just check that the result is computed (even if slowly) and is close enough.
    assert torch.allclose(C, C_ref, atol=1e-5), "Kernel output incorrect."
