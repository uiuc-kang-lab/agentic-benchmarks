
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper to build the CUDA module
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# 1. Test that calling the kernel with a non-float32 input (e.g. double) triggers an error.
def test_input_dtype_error():
    # Use small dimensions for this test.
    M, K, N = 64, 32, 64  # note N is divisible by 4
    A = torch.randn(M, K, device="cuda", dtype=torch.double)
    B = torch.randn(K, N, device="cuda", dtype=torch.double)
    
    my_module = build_kernel()
    with pytest.raises(RuntimeError):
        # The kernel assumes float32. The reinterpret cast will be invalid.
        C = my_module.forward(A, B)
        torch.cuda.synchronize()  # Force any asynchronous error

# 2. Test that a matrix B with non-16-byte-aligned memory (simulated via slicing) produces a wrong result.
def test_B_alignment_issue():
    # Create a float32 tensor that (likely) is well aligned...
    M, K, N = 64, 32, 64  # N divisible by 4
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    # Allocate a larger tensor and slice to force a misaligned pointer.
    B_full = torch.randn(K, N + 1, device="cuda", dtype=torch.float32)
    B = B_full[:, 1:]  # likely misaligns the starting address of each row
    my_module = build_kernel()
    # Run the kernel. It might produce an incorrect result if misalignment causes issues.
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    # Use a loose tolerance because misaligned loads may yield wrong results.
    assert not torch.allclose(C, C_ref, atol=1e-5), "Kernel unexpectedly produced correct results with misaligned B."

# 3. Test that an input where N is not divisible by 4 triggers an error.
def test_N_not_divisible_by_4():
    M, K, N = 64, 32, 62  # 62 is not divisible by 4
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    my_module = build_kernel()
    with pytest.raises(RuntimeError):
        C = my_module.forward(A, B)
        torch.cuda.synchronize()

# 4. Test that if the kernel launch fails (e.g., by using very large dimensions), the error is detected.
def test_kernel_launch_error():
    # We pick dimensions so huge that the grid or block dims may be invalid.
    M, K, N = 1<<15, 32, 1<<15  # large dimensions as in the provided model
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    
    my_module = build_kernel()
    # There is no explicit error checking in the kernel launch,
    # but we can force a launch error by using dimensions that exceed the hardware limits.
    # Here we simulate such a case by relying on an expected RuntimeError.
    with pytest.raises(RuntimeError):
        C = my_module.forward(A, B)
        torch.cuda.synchronize()
