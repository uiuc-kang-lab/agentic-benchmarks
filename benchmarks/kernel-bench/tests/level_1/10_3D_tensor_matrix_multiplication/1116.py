
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper to (re)build the CUDA kernel module.
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1: Batch dimension grid-limit.
# We try to simulate a large batch (N) that exceeds the typical grid dimension limit.
# In practice such a large tensor may not be allocable but we try a moderate value that is near
# the limit (if possible) and check for kernel launch errors.
def test_large_batch_size():
    my_module = build_kernel()

    # Use a batch size near the CUDA gridDim.z limit.
    # Note: Most GPUs have a limit of 65535 in the z-dimension.
    # We deliberately allocate a tensor with N = 70000 if memory allows (it may throw OOM).
    # If allocation fails, we skip the test.
    N = 70000
    M, K, L = 1, 4, 4  # set M, K, L small to stress the grid-dim limit in z.
    try:
        A = torch.randn(N, M, K, device="cuda", dtype=torch.float32)
    except RuntimeError:
        pytest.skip("Skipping test_large_batch_size due to inability to allocate huge batch tensor.")
    B = torch.randn(K, L, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        # Expecting that the kernel launch fails because gridDim.z is too high.
        C = my_module.forward(A, B)
        # Force synchronization to catch launch errors.
        torch.cuda.synchronize()

# Issue 2: Nondivisible dimensions.
# When M or L are not a multiple of 32, the kernel will launch extra threads.
# This may not cause a numerical error but might be an inefficiency.
# For our test we check that the kernel output is still correct (if not, then indexing is off).
def test_nondivisible_dimensions():
    my_module = build_kernel()
    N = 2
    M = 33  # not a multiple of 32
    K = 16
    L = 33  # not a multiple of 32
    A = torch.randn(N, M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, L, device="cuda", dtype=torch.float32)
    C = my_module.forward(A, B)
    # Synchronize and compare with torch.matmul.
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    assert torch.allclose(C, C_ref, atol=1e-5), f"Mismatch with nondivisible dimensions. Max diff: {(C - C_ref).abs().max()}"

# Issue 3: Lack of support for extra dimensions / broadcasting.
# The kernel only supports a 3D tensor (A) and a 2D matrix (B).
# Here we deliberately pass an A with 4 dimensions and B with 3 dimensions and expect the kernel to error.
def test_extra_dimensions():
    my_module = build_kernel()
    # Create a 4D tensor for A and 3D tensor for B
    A = torch.randn(2, 3, 4, 5, device="cuda", dtype=torch.float32)
    B = torch.randn(4, 5, 6, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        # We expect an error because the kernel indexing expects a 3D tensor for A.
        C = my_module.forward(A, B)
        torch.cuda.synchronize()

# Issue 4: Non-contiguous inputs.
# The kernel (via the C++ interface macros) expects the input tensors to be contiguous.
# Here we create noncontiguous versions and expect an exception due to CHECK_INPUT.
def test_non_contiguous_inputs():
    my_module = build_kernel()
    N, M, K, L = 2, 4, 6, 8
    A = torch.randn(N, M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, L, device="cuda", dtype=torch.float32)

    # Make A and B non-contiguous by transposing one of the dimensions
    A_noncontig = A.transpose(1, 2)
    B_noncontig = B.transpose(0, 1)

    # Test noncontiguous A.
    with pytest.raises(RuntimeError):
        C = my_module.forward(A_noncontig, B)
        torch.cuda.synchronize()

    # Test noncontiguous B.
    with pytest.raises(RuntimeError):
        C = my_module.forward(A, B_noncontig)
        torch.cuda.synchronize()
