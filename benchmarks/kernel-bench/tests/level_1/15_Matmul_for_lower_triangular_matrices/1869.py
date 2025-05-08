
import pytest
import torch
from torch.utils.cpp_extension import load
import os

def build_kernel():
    # Force rebuild every time to pick up changes in kernel.cu
    cuda_module = load(
        name="test_module",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Kernel only supports float32. The following test passes double tensors.
def test_non_float32_input():
    my_module = build_kernel()
    N = 128
    # create double tensors (64-bit) instead of float32
    A = torch.randn(N, N, dtype=torch.double, device="cuda")
    B = torch.randn(N, N, dtype=torch.double, device="cuda")
    # Even though the kernel expects float, it will run and produce incorrect results.
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    # Compute the expected result in double precision and then take the tril
    C_ref = torch.tril(torch.matmul(A, B))
    # The results are expected to be very different or inaccurate because memory is misinterpreted.
    assert not torch.allclose(C.double(), C_ref, atol=1e-5), (
        "Kernel did not fail on double input - it should only support float32."
    )

# Issue 2: Lack of proper synchronization can delay error detection.
# Here we induce an error by creating an input that leads to an out‐of-bound access.
def test_no_device_synchronize_error_detection():
    my_module = build_kernel()
    # We deliberately use a very small matrix which may cause an issue in a more complex scenario.
    # (Since the kernel mapping is row-based, forcing an erroneous launch is not trivial.
    #  Instead, we simulate a situation where the asynchronous error might be hidden.)
    N = 0  # Passing an empty matrix should ideally trigger error checks in a robust implementation.
    A = torch.empty(0, 0, device="cuda", dtype=torch.float32)
    B = torch.empty(0, 0, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        # If the kernel encounters an asynchronous error, proper synchronization would raise an error.
        C = my_module.forward(A, B)
        torch.cuda.synchronize()

# Issue 3: Kernel does not support batched or non-contiguous inputs.
def test_batched_input_not_supported():
    my_module = build_kernel()
    N = 64
    # Create a batched input (3D tensor) which should not be accepted.
    A = torch.tril(torch.randn(2, N, N, device="cuda", dtype=torch.float32))
    B = torch.tril(torch.randn(2, N, N, device="cuda", dtype=torch.float32))
    with pytest.raises(RuntimeError):
        # The kernel expects a 2D matrix. The TORCH_CHECK(A.dim() == 2) should trigger.
        my_module.forward(A, B)

# Issue 4: Fixed warp partitioning may be inefficient (or subtly wrong)
# for matrices where dimensions are not multiples of the warp size.
# We use a matrix dimension that is not a multiple of 32 and compare to reference.
def test_incorrect_warp_partitioning():
    my_module = build_kernel()
    N = 70  # Not a multiple of warp size (32)
    A = torch.tril(torch.randn(N, N, device="cuda", dtype=torch.float32))
    B = torch.tril(torch.randn(N, N, device="cuda", dtype=torch.float32))
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.tril(torch.matmul(A, B))
    # Due to potential indexing or partitioning inefficiencies, the result may be wrong.
    # We assert that the outputs are not close, indicating the kernel did not handle this case properly.
    assert not torch.allclose(C, C_ref, atol=1e-4), (
        f"Kernel output is unexpectedly correct for a matrix dimension (N={N}) that is not a multiple of the warp size."
    )
