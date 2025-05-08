
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test Case 1: Trigger issue with wrong data type (double input) because the kernel expects float32.
def test_wrong_dtype():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    M = 128
    K = 64
    N = 128
    # Create double precision tensors.
    A = torch.randn(M, K, dtype=torch.float64, device="cuda")
    B = torch.randn(K, N, dtype=torch.float64, device="cuda")
    module = build_kernel()
    # Calling the CUDA kernel with double tensors will misuse the data_ptr<float>() casts.
    # We expect the result to differ significantly from torch.matmul.
    C_kernel = module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    # The result is not expected to match because of precision reinterpretation.
    assert not torch.allclose(C_kernel.to(torch.float64), C_ref, atol=1e-5), \
        "Kernel incorrectly handled double precision input, but it should only support float32."

# Test Case 2: Trigger issue with batched input since the kernel only supports 2D matrices.
def test_batched_input():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    batch = 4
    M = 64
    K = 32
    N = 64
    # Create batched tensors (3D) which the kernel does not expect.
    A = torch.randn(batch, M, K, dtype=torch.float32, device="cuda")
    B = torch.randn(batch, K, N, dtype=torch.float32, device="cuda")
    module = build_kernel()
    # Flattening the batch dimension, a naive call to the kernel would mistakenly treat the first batch as rows.
    # We expect the output to be incorrect compared to torch.matmul performed batchwise.
    # Here, we simply demonstrate that the kernel cannot handle batched inputs.
    with pytest.raises(RuntimeError):
        # The module expects 2D tensors; passing 3D tensors should trigger an error (or produce wrong results).
        C_kernel = module.forward(A, B)
        torch.cuda.synchronize()

# Test Case 3: Trigger issue with half-precision input.
def test_half_precision_input():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    M = 128
    K = 64
    N = 128
    # Create half precision (float16) tensors.
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(K, N, dtype=torch.float16, device="cuda")
    module = build_kernel()
    # Even if the tensors are half precision, the kernel will call data_ptr<float>()
    # causing misinterpretation of the data bits.
    C_kernel = module.forward(A, B)
    torch.cuda.synchronize()
    # We compute a reference result using torch.matmul after converting A and B to float.
    C_ref = torch.matmul(A.float(), B.float()).half()
    # The results are likely wrong due to type misinterpretation.
    assert not torch.allclose(C_kernel, C_ref, atol=1e-2), \
        "Kernel incorrectly handled float16 input, but it should only support float32."
