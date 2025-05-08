
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

# Test 1: Pass non-float32 tensors (e.g., double). 
# This should trigger an issue since the kernel is hard-coded to use float.
def test_non_float_tensor():
    N = 64
    A = torch.randn(N, N, device="cuda", dtype=torch.double)
    B = torch.randn(N, N, device="cuda", dtype=torch.double)
    my_module = build_kernel()
    with pytest.raises(Exception):
        # Expecting an error or incorrect behavior because the pointer types are float* in the kernel.
        C = my_module.forward(A, B)
        torch.cuda.synchronize()

# Test 2: Pass non-contiguous tensors.
# The kernel uses CHECK_CONTIGUOUS to ensure inputs are contiguous. 
def test_non_contiguous_tensor():
    N = 64
    A = torch.randn(N, N, device="cuda", dtype=torch.float32).t()  # Transpose usually makes tensor non-contiguous
    B = torch.randn(N, N, device="cuda", dtype=torch.float32).t()
    my_module = build_kernel()
    with pytest.raises(Exception):
        # The CHECK_INPUT macros should trigger an error for non-contiguous tensors.
        C = my_module.forward(A, B)
        torch.cuda.synchronize()

# Test 3: Pass batched inputs (3D tensors) where the kernel expects 2D matrices.
# This should trigger an error in dimension handling.
def test_batched_input():
    batch = 4
    M = 32
    K = 16
    N = 24
    # Create batched matrices. The kernel assumes A is 2D, so using 3D file input should cause a failure.
    A = torch.randn(batch, M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(batch, K, N, device="cuda", dtype=torch.float32)
    my_module = build_kernel()
    with pytest.raises(Exception):
        # Since the kernel simply calls size(0) and size(1), with batched input the dimensions are wrong.
        C = my_module.forward(A, B)
        torch.cuda.synchronize()

# Test 4: Although not directly used, the existence of an unused transpose kernel may indicate integration problems.
# Here we try to indirectly trigger its effect by checking that performing a matmul with non-transposed inputs
# (when perhaps a transpose was expected in a more general implementation) does not match torch.matmul.
def test_output_mismatch_due_to_memory_layout_assumption():
    # This test is more advisory: if the user intended a general case, the current implementation works only
    # under strict assumptions. We check that if we swap dimensions improperly, we do not get the expected result.
    M = 64
    K = 32
    N = 48
    A = torch.randn(M, K, device="cuda", dtype=torch.float32).contiguous()
    B = torch.randn(K, N, device="cuda", dtype=torch.float32).contiguous()
    my_module = build_kernel()
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    # If the kernel were generalized (and perhaps using a transpose kernel to correct layout),
    # the outputs would match. Here we deliberately assert they should differ to catch assumptions.
    assert not torch.allclose(C, C_ref, atol=1e-5), "Kernel output unexpectedly matches the reference output despite layout assumptions."
