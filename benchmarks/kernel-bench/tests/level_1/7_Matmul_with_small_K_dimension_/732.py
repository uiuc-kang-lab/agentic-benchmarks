
import torch
import pytest
from torch.utils.cpp_extension import load

# Helper function to build the CUDA extension.
def build_kernel():
    return load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

# Test case to trigger issues 1 and 2:
# When using small matrix sizes the kernel path using constant memory is selected.
# Since writing to __constant__ memory from the device is not allowed and shared memory should be used,
# we expect the result to significantly differ from torch.matmul.
def test_constant_memory_kernel():
    M, K, N = 128, 32, 128  # small dimensions to trigger constant memory version
    A = torch.randn(M, K, device="cuda", dtype=torch.float32).contiguous()
    B = torch.randn(K, N, device="cuda", dtype=torch.float32).contiguous()
    my_module = build_kernel()
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    # We expect the result to be off due to the misuse of constant memory.
    assert not torch.allclose(C, C_ref, atol=1e-3), \
        f"Constant memory kernel produced a result that matches torch.matmul; expected undefined behavior."

# Test case to trigger issue 3:
# The kernel does not verify that input tensors are of type float32.
# Passing double precision tensors (float64) may produce incorrect results.
def test_input_tensor_type():
    M, K, N = 128, 32, 128  # use small dimensions
    A = torch.randn(M, K, device="cuda", dtype=torch.float64).contiguous()
    B = torch.randn(K, N, device="cuda", dtype=torch.float64).contiguous()
    my_module = build_kernel()
    # Since the kernel expects float32 pointers, the double input is misinterpreted.
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    # We expect these results to diverge as types are mismatched.
    assert not torch.allclose(C.to(torch.float64), C_ref, atol=1e-3), \
        "Kernel incorrectly accepted non-float32 input; result unexpectedly matches reference."

# Test case to trigger issue 4:
# For large matrices the cublasSgemm path is used.
# Due to the incorrect configuration for row-major layout,
# the computed result will be transposed (or entirely incorrect) compared to torch.matmul.
def test_cublas_call_transposition():
    # use dimensions greater than MATRIX_SIZE_THRESHOLD so that the cublasSgemm path is taken.
    M, K, N = 1024, 64, 1024  # M and N are > 512 threshold in our kernel condition
    A = torch.randn(M, K, device="cuda", dtype=torch.float32).contiguous()
    B = torch.randn(K, N, device="cuda", dtype=torch.float32).contiguous()
    my_module = build_kernel()
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    # We expect the result from cuBLAS to be wrong due to the assumed row-major ordering issues.
    assert not torch.allclose(C, C_ref, atol=1e-3), \
        "cuBLAS call returned a result matching torch.matmul; expected transposition/layout error."

if __name__ == "__main__":
    pytest.main([__file__])
