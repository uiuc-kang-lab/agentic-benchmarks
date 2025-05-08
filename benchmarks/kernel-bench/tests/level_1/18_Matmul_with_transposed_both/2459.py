
import pytest
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

# Issue 1: Kernel does not support batched operations.
# Create an input with an extra batch dimension.
def test_batched_input():
    # Create batched tensors that are 3D instead of 2D.
    batch = 4
    K = 32
    M = 16
    N = 24
    A = torch.randn(batch, K, M, device="cuda", dtype=torch.float32)
    B = torch.randn(batch, N, K, device="cuda", dtype=torch.float32)
    my_module = build_kernel()
    # Expect that the kernel, which is written for 2D inputs, either produces
    # an error or produces an output of unexpected shape.
    with pytest.raises(Exception):
        # We simulate that the kernel should not accept batched inputs.
        C = my_module.forward(A, B)
        torch.cuda.synchronize()

# Issue 2: Kernel does not handle non-contiguous inputs.
def test_non_contiguous_inputs():
    # Create contiguous inputs with expected shapes.
    K = 64
    M = 32
    N = 48
    A = torch.randn(K, M, device="cuda", dtype=torch.float32)
    B = torch.randn(N, K, device="cuda", dtype=torch.float32)
    # Make non-contiguous versions by transposing twice (or by slicing).
    A_noncontig = A.t().t()  # this is a no-op in terms of data but can result in non-contiguous memory
    B_noncontig = B.t().t()
    # Ensure they are non-contiguous.
    if A_noncontig.is_contiguous():
        A_noncontig = A_noncontig[:, ::2].repeat(1, 2)
    if B_noncontig.is_contiguous():
        B_noncontig = B_noncontig[:, ::2].repeat(1, 2)
    my_module = build_kernel()
    # The kernel does not check for contiguity so output may be wrong.
    C = my_module.forward(A_noncontig, B_noncontig)
    torch.cuda.synchronize()
    # Compute reference using torch.matmul on the transposed tensors.
    C_ref = torch.matmul(A_noncontig.t(), B_noncontig.t())
    # We expect a mismatch between C and the reference due to unexpected memory access.
    assert not torch.allclose(C, C_ref, atol=1e-5), \
        "Kernel unexpectedly handled non-contiguous inputs correctly."

# Issue 3: Lack of error checking for CUDA kernel launch errors.
def test_kernel_error_checking():
    # Forcing an error condition via extreme grid dimensions.
    # We provide inputs that are extremely large so that grid size overflows.
    # Note: This test case is artificial since in realistic use cases, dimensions come from the user.
    K = 1024
    M = 1 << 16  # Large dimension to force grid dim overflow
    N = 1 << 16
    try:
        A = torch.randn(K, M, device="cuda", dtype=torch.float32)
        B = torch.randn(N, K, device="cuda", dtype=torch.float32)
    except RuntimeError:
        pytest.skip("CUDA memory not sufficient to allocate extremely large tensors.")

    my_module = build_kernel()
    # In many cases, the kernel launch will fail silently if an error check is absent.
    # Here we expect that an error will raise upon synchronization.
    C = my_module.forward(A, B)
    with pytest.raises(RuntimeError):
        torch.cuda.synchronize()
