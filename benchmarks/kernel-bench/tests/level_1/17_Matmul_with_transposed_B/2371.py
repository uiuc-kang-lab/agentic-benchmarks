
import pytest
import torch
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

# Test case 1: Use an input tensor of type double to trigger the type incompatibility issue.
def test_input_tensor_type():
    # Create double precision tensors (should be float32 for the kernel)
    M, K, N = 64, 64, 64
    A = torch.randn(M, K, dtype=torch.double, device="cuda")
    # Note: B should have shape (N, K) to match kernel expectation (B will be transposed internally)
    B = torch.randn(N, K, dtype=torch.double, device="cuda")
    
    my_module = build_kernel()
    # Expect the kernel to produce wrong results since it assumes float32
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    # Compute reference result using the same operation after manually converting B.T back
    C_ref = torch.matmul(A, B.T)
    # The error should be huge because of type misuse (or simply not close)
    assert not torch.allclose(C.double(), C_ref, atol=1e-5), (
        "Kernel unexpectedly produced correct results for double inputs. "
        "This should fail because the kernel only supports float32."
    )

# Test case 2: Use a batched input (3D tensor) to trigger the 2D-only implementation issue.
def test_batched_input():
    # Create batched tensors (e.g. batch_size = 4)
    BATCH, M, K, N = 4, 32, 32, 32
    A = torch.randn(BATCH, M, K, device="cuda", dtype=torch.float32)
    # Note: B should be of shape (BATCH, N, K) to be transposed later to (BATCH, K, N)
    B = torch.randn(BATCH, N, K, device="cuda", dtype=torch.float32)
    
    my_module = build_kernel()
    with pytest.raises(RuntimeError, match="A must be 2D"):
        # The C++ TORCH_CHECK in forward expects 2D tensors, so passing batched tensors should trigger an error.
        my_module.forward(A, B)

# Test case 3: Test a potential issue with the assumption on the layout of B.
def test_incorrect_b_layout():
    # Here we mimic a scenario where B is provided already transposed,
    # meaning its shape is (K, N) rather than the expected (N, K).
    M, K, N = 64, 64, 32
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    # Create B in transposed layout: expected layout for kernel is (N, K)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    
    my_module = build_kernel()
    # The kernel will treat B as if it has shape (N, K), which is incorrect in this test.
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    # Compute correct result manually using B transposed back
    C_ref = torch.matmul(A, B)
    # Expect significant differences because the kernel's assumption on B's layout is violated.
    assert not torch.allclose(C, C_ref, atol=1e-5), (
        "Kernel produced results close to the reference despite an incorrect B layout. "
        "This test should detect the hard-coded assumption on B’s memory layout."
    )
