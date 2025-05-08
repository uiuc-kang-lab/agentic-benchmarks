
import pytest
import torch
from torch.utils.cpp_extension import load
from torch.testing import assert_allclose

def build_kernel():
    # Build and load the CUDA extension module from kernel.cu.
    return load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

# Issue 1: Data type limitation.
# The kernel only supports float32. Passing double tensors causes misinterpretation,
# leading to incorrect results (or at least results that are not almost equal to the reference).
def test_data_type_mismatch():
    cuda_module = build_kernel()
    M, K, N = 128, 64, 128
    # Create double precision tensors.
    A = torch.randn(M, K, dtype=torch.double, device="cuda")
    B = torch.randn(N, K, dtype=torch.double, device="cuda")
    # Convert to double gold result using torch.matmul on A and B.
    C_ref = torch.matmul(A, B.T)
    # Call our custom kernel.
    # Although the kernel is not type-checked at the C++ level, it assumes float.
    C = cuda_module.forward(A, B)
    torch.cuda.synchronize()
    # The output is expected to be wrong because the underlying kernel interprets the memory as float32.
    with pytest.raises(AssertionError):
        assert_allclose(C, C_ref, atol=1e-5)

# Issue 2: Inconsistent input shapes.
# The Python documentation states that B should be (K, N) but the implementation (and get_inputs)
# expects B to be (N, K). This test deliberately passes B with shape (K, N) to trigger a shape mismatch.
def test_shape_mismatch():
    cuda_module = build_kernel()
    M, K, N = 64, 128, 256
    # Create A with shape (M, K)
    A = torch.randn(M, K, dtype=torch.float32, device="cuda")
    # Incorrect shape for B: using (K, N) instead of (N, K)
    B = torch.randn(K, N, dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError) as excinfo:
        cuda_module.forward(A, B)
    # Check that the error refers to the K-dimension mismatch enforced by TORCH_CHECK.
    assert "A and B must have the same K dimension" in str(excinfo.value)

# Issue 3 and Issue 4 are performance/optimization concerns.
# They may not break correctness but can be triggered by using matrix sizes that
# are not multiples of the fixed block dimensions. This test uses such sizes and checks correctness.
def test_non_divisible_by_block_size():
    cuda_module = build_kernel()
    # Choose dimensions that are not multiples of 16.
    M, K, N = 30, 17, 25
    A = torch.randn(M, K, dtype=torch.float32, device="cuda")
    B = torch.randn(N, K, dtype=torch.float32, device="cuda")
    C = cuda_module.forward(A, B)
    torch.cuda.synchronize()
    # The reference computation (note: B must be transposed for the multiplication)
    C_ref = torch.matmul(A, B.T)
    assert_allclose(C, C_ref, atol=1e-5)
