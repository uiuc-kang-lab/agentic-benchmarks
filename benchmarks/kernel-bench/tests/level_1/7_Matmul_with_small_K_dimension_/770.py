
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

# Issue 1: Input tensor type is not enforced.
# When passing tensors that are not float32, the kernel will treat the underlying data incorrectly.
def test_input_dtype_mismatch():
    # Use a non-float32 type (double) so that data_ptr<float>() misinterprets the data.
    M, K, N = 64, 64, 64  # Use sizes that trigger the custom kernel.
    A = torch.randn(M, K, dtype=torch.float64, device="cuda").contiguous()
    B = torch.randn(K, N, dtype=torch.float64, device="cuda").contiguous()
    mod = build_kernel()
    # The kernel does not check for dtype, so it will run but produce wrong outputs.
    C = mod.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A.float(), B.float())
    # The result should be very different (or at least not close) because of dtype misinterpretation.
    assert not torch.allclose(C, C_ref, atol=1e-3), "Kernel unexpectedly produced correct result with non-float32 input."

# Issue 2: No error checking after kernel launch.
# We simulate a scenario that forces the custom kernel path.
# If the custom kernel had a launch error, it might go unnoticed.
def test_kernel_launch_error_detection():
    M, K, N = 32, 32, 32  # Small sizes will force custom kernel path.
    A = torch.randn(M, K, dtype=torch.float32, device="cuda").contiguous()
    B = torch.randn(K, N, dtype=torch.float32, device="cuda").contiguous()
    mod = build_kernel()
    # Run the forward pass.
    C = mod.forward(A, B)
    # Not checking cudaGetLastError, so if there was a launch error it would go unnoticed,
    # but we can compare to torch.matmul to see if the result is off.
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    # Expect differences if a launch error occurred.
    if not torch.allclose(C, C_ref, atol=1e-3):
        pytest.skip("Detected potential kernel launch issue (no cudaGetLastError check).")
    else:
        pytest.fail("Kernel launch error went undetected; custom kernel did not trigger any error.")

# Issue 3: Lack of cuBLAS error checking.
# We trigger this by giving incorrectly-sized matrices that force the cuBLAS branch.
def test_cublas_error_checking():
    # Create matrices with mismatched dimensions.
    # A is (M, K) and B is (K_wrong, N) with K_wrong != K.
    M, K, N = 128, 128, 128  # Choose sizes to trigger cuBLAS branch (all dimensions >= 64).
    A = torch.randn(M, K, dtype=torch.float32, device="cuda").contiguous()
    # B should be (K, N), but we deliberately make it (K-1, N)
    B = torch.randn(K - 1, N, dtype=torch.float32, device="cuda").contiguous()
    mod = build_kernel()
    with pytest.raises(RuntimeError):
        # Expect an error due to incompatible dimensions leading to an out-of-bound memory access in cuBLAS call.
        C = mod.forward(A, B)
        torch.cuda.synchronize()

# Issue 4: Assumption of contiguous input.
# We pass noncontiguous tensors and expect the CHECK_CONTIGUOUS macro to raise an error.
def test_noncontiguous_input():
    M, K, N = 128, 128, 128  # Use sizes that trigger cuBLAS branch.
    A_full = torch.randn(M, K, dtype=torch.float32, device="cuda")
    B_full = torch.randn(K, N, dtype=torch.float32, device="cuda")
    # Create noncontiguous versions by transposing
    A = A_full.t()  # Transposed tensor is noncontiguous
    B = B_full.t()  # Noncontiguous
    mod = build_kernel()
    with pytest.raises(RuntimeError):
        mod.forward(A, B)
