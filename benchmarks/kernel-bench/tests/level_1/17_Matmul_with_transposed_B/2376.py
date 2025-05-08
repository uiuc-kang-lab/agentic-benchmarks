
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

# Issue 1: Kernel only supports float32.
def test_non_float_dtype():
    my_module = build_kernel()
    M, K, N = 64, 32, 48
    A = torch.randn(M, K, dtype=torch.double, device="cuda")  # double type
    B = torch.randn(N, K, dtype=torch.double, device="cuda")
    with pytest.raises(RuntimeError, match=""):
        # Expect a type check assertion inside the kernel or host code to fail
        my_module.forward(A, B)

# Issue 2: Kernel does not support batched inputs.
def test_batched_input_error():
    my_module = build_kernel()
    # Create a batched 3D tensor for A and B
    batch = 4
    M, K, N = 16, 32, 24
    A = torch.randn(batch, M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(batch, N, K, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match=""):
        # The kernel expects 2D inputs only.
        my_module.forward(A, B)

# Issue 3: Kernel assumes inputs are contiguous.
def test_non_contiguous_input_error():
    my_module = build_kernel()
    M, K, N = 64, 32, 48
    A_full = torch.randn(K, M, device="cuda", dtype=torch.float32)  # intentionally swapped dimensions
    B_full = torch.randn(K, N, device="cuda", dtype=torch.float32)
    # Transpose to get correct shape but non-contiguous layout.
    A = A_full.t()  # shape (M, K) but non-contiguous
    B = B_full.t()  # shape (N, K) but non-contiguous
    with pytest.raises(RuntimeError, match="contiguous"):
        my_module.forward(A, B)

# Issue 4: Handling non-divisible dimensions (boundary conditions).
def test_non_divisible_dimensions():
    my_module = build_kernel()
    # Dimensions not divisible by BLOCK_SIZE (16) or BLOCK_SIZE*COARSENING (32)
    M, K, N = 30, 37, 29
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(N, K, device="cuda", dtype=torch.float32)
    C = my_module.forward(A, B)
    # The expected result: A * B^T, i.e., shape (M, N)
    C_ref = torch.matmul(A, B.t())
    torch.cuda.synchronize()
    assert torch.allclose(C, C_ref, atol=1e-4), f"Output does not match reference for non-divisible dims. Max diff: {(C-C_ref).abs().max()}"

# Issue 5: Potential uncoalesced memory accesses may not directly trigger an error,
# but for testing we can measure performance on a larger input to see degradation.
# This test is a best‐effort performance check (it does not enforce a pass/fail)
def test_large_input_performance():
    my_module = build_kernel()
    M, K, N = 1024, 1024, 1024
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(N, K, device="cuda", dtype=torch.float32)
    # We run the kernel multiple times and just ensure it completes correctly.
    for _ in range(10):
        C = my_module.forward(A, B)
    torch.cuda.synchronize()
    # Check one output to ensure computation is done (not a performance metric).
    C_ref = torch.matmul(A, B.t())
    assert torch.allclose(C, C_ref, atol=1e-3), "Large matrix multiplication produced unexpected output."

# Issue 6: Kernel may hide asynchronous errors because of missing cudaDeviceSynchronize() after launch.
def test_async_error_detection():
    my_module = build_kernel()
    # Provide an input where M or K is zero, which might trigger unusual behavior.
    M, K, N = 0, 32, 48
    A = torch.empty(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(N, K, device="cuda", dtype=torch.float32)
    # The kernel should either handle the empty tensor gracefully or trigger an error.
    try:
        C = my_module.forward(A, B)
        # Ensure that the resulting tensor has the expected shape.
        assert list(C.shape) == [M, N], "Output shape incorrect for empty input."
    except RuntimeError as e:
        pytest.fail("Kernel raised a runtime error on empty input, possibly hiding async errors.")

if __name__ == '__main__':
    pytest.main([__file__])
