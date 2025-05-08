
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

# Issue 1: Unused reduction kernel.
# There is no direct interface to the reduce kernel.
# We simulate an indirect test by checking that calling the forward (i.e. matmul) function does not inadvertently use or expose reduction functionality.
def test_unused_reduction_kernel():
    my_module = build_kernel()
    # Since there's no API to call the reduction kernel separately,
    # we simply ensure that the extension only exposes 'forward'.
    assert hasattr(my_module, "forward"), "Expected module to expose a 'forward' function."
    # If additional functions were unintentionally exposed, that might be an issue.
    # (For the test case, we assume that 'reduce' should not be exposed.)
    exposed_functions = [attr for attr in dir(my_module) if not attr.startswith("_")]
    assert "reduce" not in exposed_functions, "Reduction kernel appears to be exposed, but it is not used in the matmul path."

# Issue 2: Incorrect cuBLAS parameter ordering.
def test_incorrect_gemm_parameters():
    my_module = build_kernel()
    M, K, N = 32, 16, 24
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    
    # Use our kernel (which calls cublasSgemm) to compute C.
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    
    # Compute the expected result using torch.matmul.
    C_ref = torch.matmul(A, B)
    # Because of the likely misordering of GEMM parameters, the result may be transposed or completely wrong.
    # We deliberately check for a mismatch.
    if torch.allclose(C, C_ref, atol=1e-3):
        pytest.fail("cuBLAS parameter misordering: Kernel output matches torch.matmul unexpectedly.")
    else:
        # The test passes if the output is different from torch.matmul.
        assert True

# Issue 3: Missing error checking for cuBLAS API calls.
# Without internal error checking the user may not receive an informative error for unsupported types.
def test_cublas_error_on_unsupported_dtype():
    my_module = build_kernel()
    M, K, N = 32, 16, 24
    # Create double-precision tensors.
    A = torch.randn(M, K, device="cuda", dtype=torch.float64)
    B = torch.randn(K, N, device="cuda", dtype=torch.float64)
    with pytest.raises(RuntimeError):
        # The CHECK_INPUT macro in the kernel only checks for CUDA and contiguous, not dtype.
        # The cuBLAS call (cublasSgemm) will not execute correctly on double tensors.
        my_module.forward(A, B)

# Issue 4: Kernel only handles float32 inputs.
def test_non_float32_input():
    my_module = build_kernel()
    M, K, N = 32, 16, 24
    # Create integer tensors which are certainly unsupported.
    A = torch.randint(0, 10, (M, K), device="cuda", dtype=torch.int32)
    B = torch.randint(0, 10, (K, N), device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError):
        my_module.forward(A, B)

# Issue 5: Misleading documentation regarding shared memory and warp-level optimization
# Since the shared memory apply only to the unused reduction kernel,
# we insert a test to check that no unexpected shared memory optimization is applied to the output.
def test_shared_memory_claim():
    my_module = build_kernel()
    M, K, N = 64, 32, 48
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    # Compute reference output.
    C_ref = torch.matmul(A, B)
    # In a correct, optimized implementation the output should match.
    # However, if the shared memory / warp-level optimization was correctly applied,
    # then even using reduction kernel techniques the result should match exactly.
    # Since our reduction kernel is not used in the computation, a discrepancy will indicate an issue.
    if not torch.allclose(C, C_ref, atol=1e-3):
        pytest.fail("Output does not match torch.matmul suggesting misapplied shared memory optimizations.")
