
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper to build the CUDA kernel module from kernel.cu
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1: Misleading shared memory comment
# (While we cannot directly verify shared memory usage from Python,
#  we can check that the kernel produces results that match torch.matmul.
#  In a real-world setting, one might use profiling tools to check for shared memory usage.)
def test_shared_memory_comment_is_misleading():
    my_module = build_kernel()
    N, M, K, L = 4, 8, 16, 10
    # Use a moderate size so that if shared memory were used, it might allocate some.
    A = torch.randn(N, M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, L, device="cuda", dtype=torch.float32)
    output = my_module.forward(A, B)
    ref = torch.matmul(A, B)
    assert torch.allclose(output, ref, atol=1e-5), \
        f"Kernel output does not match torch.matmul output. Max diff: {(output - ref).abs().max().item()}"

# Issue 2: Lack of support for broadcasting / non-3D input for A
def test_input_dimension_broadcasting_failure():
    my_module = build_kernel()
    # Pass a 2D tensor for A, which should not be accepted by our specialized kernel
    M, K, L = 8, 16, 10
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)  # 2D tensor instead of 3D
    B = torch.randn(K, L, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        my_module.forward(A, B)

# Issue 2 (alternative): Pass an input that has extra dimensions
def test_input_dimension_extra_dims_failure():
    my_module = build_kernel()
    # Pass a 4D tensor for A which our kernel does not expect.
    N, M, K, L = 2, 4, 8, 10
    A = torch.randn(1, N, M, K, device="cuda", dtype=torch.float32)  # extra dimension at the front
    B = torch.randn(K, L, device="cuda", dtype=torch.float32)
    with pytest.raises(IndexError):
        # The kernel accesses A assuming 3D shape; using extra dims may lead to indexing error.
        my_module.forward(A, B)

# Issue 3: Fixed block dimensions might cause subtle misbehavior with very small matrices.
def test_fixed_block_dimensions_small_matrices():
    my_module = build_kernel()
    # Use matrix sizes that are smaller than the block dimensions; the kernel still should work correctly,
    # but this test demonstrates that the kernel may be over-provisioned.
    N, M, K, L = 1, 5, 7, 3  # Small sizes; 5 < 32 and 3 < 32.
    A = torch.randn(N, M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, L, device="cuda", dtype=torch.float32)
    output = my_module.forward(A, B)
    ref = torch.matmul(A, B)
    assert torch.allclose(output, ref, atol=1e-5), \
        f"Kernel output on small matrices differs from torch.matmul. Max diff: {(output - ref).abs().max().item()}"

# Issue 4: Loop unrolling on non-constant K.
def test_loop_unroll_on_non_constant_K():
    my_module = build_kernel()
    # Vary K at runtime; although the kernel will work, the unroll pragma is not beneficial.
    # Here we simply verify that for different K, the kernel returns the correct output.
    N = 2
    M = 10
    # Test several values of K to simulate non-constant loop depths.
    for K in [13, 27, 64]:
        L = 8
        A = torch.randn(N, M, K, device="cuda", dtype=torch.float32)
        B = torch.randn(K, L, device="cuda", dtype=torch.float32)
        output = my_module.forward(A, B)
        ref = torch.matmul(A, B)
        assert torch.allclose(output, ref, atol=1e-5), \
            f"For K={K}, kernel output differs from torch.matmul. Max diff: {(output-ref).abs().max().item()}"

# Extra test: Kernel only supports floating point tensors. Passing an integer tensor should raise an error.
def test_invalid_dtype():
    my_module = build_kernel()
    N, M, K, L = 4, 8, 16, 10
    A = torch.randint(low=0, high=10, size=(N, M, K), device="cuda", dtype=torch.int32)
    B = torch.randint(low=0, high=10, size=(K, L), device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError):
        my_module.forward(A, B)

# Extra test: Non-contiguous tensors should be rejected by the input checks.
def test_non_contiguous_input():
    my_module = build_kernel()
    N, M, K, L = 4, 8, 16, 10
    A = torch.randn(N, M, K, device="cuda", dtype=torch.float32).transpose(1,2)  # make it non-contiguous
    B = torch.randn(K, L, device="cuda", dtype=torch.float32).transpose(0,1)  # make it non-contiguous
    with pytest.raises(RuntimeError):
        my_module.forward(A, B)
