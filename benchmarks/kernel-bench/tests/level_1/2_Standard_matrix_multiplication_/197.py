
import pytest
import torch
from torch.utils.cpp_extension import load

# Function to compile and load the CUDA kernel module
def build_kernel():
    cuda_module = load(
        name="kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Misaligned access for matrix B
def test_misaligned_B():
    # Create B with an extra column and take a narrow slice to force misalignment.
    # Typically, the result of a narrow() may have an offset that makes its pointer not 16-byte aligned.
    M, K, N = 64, 128, 65  # Choose N = 65 so that later narrowing gives N divisible by 4 after slicing.
    # Ensure N is divisible by 4 for the vectorized load in the kernel.
    # Allocate a larger tensor and then slice so that the starting address is misaligned.
    B_full = torch.randn(K, N + 1, device="cuda", dtype=torch.float32)
    # Narrow B to drop the first column – this may misalign the storage pointer.
    B = B_full.narrow(1, 1, N)
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    
    # Only run if the data pointer of B is not 16-byte aligned.
    ptr = B.data_ptr()
    if ptr % 16 == 0:
        pytest.skip("B is unexpectedly 16-byte aligned, cannot test misalignment.")
    
    kernel = build_kernel()
    C = kernel.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    # If misalignment causes wrong data to be loaded, the result will differ.
    assert not torch.allclose(C, C_ref, atol=1e-5), "Misaligned load expected to produce incorrect results."

# Issue 2: Input tensor type not float32
def test_input_tensor_type():
    M, K, N = 64, 128, 64
    A = torch.randn(M, K, device="cuda", dtype=torch.float64)  # Wrong dtype: float64
    B = torch.randn(K, N, device="cuda", dtype=torch.float64)
    
    kernel = build_kernel()
    with pytest.raises(RuntimeError):
        # Should trigger error in CHECK_INPUT or produce incorrect behavior
        C = kernel.forward(A, B)
        torch.cuda.synchronize()

# Issue 3: Kernel launch error is not checked.
def test_kernel_launch_error_checking():
    # Pass in deliberately non-contiguous tensors which fail the CHECK_CONTIGUOUS macro
    M, K, N = 64, 128, 64
    A = torch.randn(M, K, device="cuda", dtype=torch.float32).t()  # Transposed makes it non-contiguous
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    
    kernel = build_kernel()
    with pytest.raises(RuntimeError):
        C = kernel.forward(A, B)
        torch.cuda.synchronize()

# Issue 4: Suboptimal fallback when N is not divisible by 4.
def test_N_not_divisible_by_4():
    # Choose N such that N % 4 != 0 to trigger the element-wise fallback in vectorized B loads.
    M, K, N = 64, 128, 62  # 62 is not divisible by 4.
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    
    kernel = build_kernel()
    C = kernel.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    # While correctness may hold, performance degradation or extra branches are expected.
    assert torch.allclose(C, C_ref, atol=1e-5), "Kernel output differs from reference for N not divisible by 4."

# Issue 5: Using __ldg() without verifying read-only properties or proper alignment can be problematic.
# We simulate a scenario where tensor A is non-contiguous to force improper use of __ldg().
def test_non_contiguous_A():
    # Create A, then perform an operation to make it non-contiguous while preserving shape.
    M, K, N = 64, 128, 64
    A_full = torch.randn(M*2, K, device="cuda", dtype=torch.float32)
    # Take every other row, which produces a non-contiguous tensor.
    A = A_full[::2, :]
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    
    kernel = build_kernel()
    with pytest.raises(RuntimeError):
        C = kernel.forward(A, B)
        torch.cuda.synchronize()
