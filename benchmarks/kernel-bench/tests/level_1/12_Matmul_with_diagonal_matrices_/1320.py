
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Utility to build/load the CUDA extension
def build_kernel():
    # Remove any previous build artifacts to force a rebuild
    build_dir = os.path.join(os.path.dirname(__file__), "build")
    cuda_module = load(
        name="diag_matmul",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
        build_directory=build_dir
    )
    return cuda_module

# Issue 1: Test that passing input tensors of type other than float32 (e.g. double) results in incorrect computations.
def test_input_tensor_type():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available, skipping test for tensor type.")
    # Create inputs of type double rather than float32.
    N = 1024
    M = 1024
    A = torch.randn(N, dtype=torch.double, device='cuda')
    B = torch.randn(N, M, dtype=torch.double, device='cuda')
    
    module = build_kernel()
    
    # Calling the kernel: the code uses A.data_ptr<float>(), so interpreting double as float.
    C_kernel = module.forward(A, B)
    # Compute reference using proper type conversion.
    C_ref = torch.diag(A.float()) @ B.float()
    
    # Since the kernel misinterprets the data type, the output should be very different
    # We check that the computed kernel output is not close to the reference.
    with pytest.raises(AssertionError):
        # If by chance they are close (extremely unlikely), we then force an assertion.
        assert torch.allclose(C_kernel, C_ref, atol=1e-3), "Kernel output is unexpectedly correct for double inputs!"

# Issue 2: Test that for very large matrix sizes the index arithmetic may overflow.
# Note: In practice such large tensors may exceed available GPU memory.
# We simulate the condition by using sizes that when multiplied exceed 32-bit integer range.
def test_index_overflow():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available, skipping test for index overflow.")
    # Check if we can simulate an overflow scenario.
    # The maximum positive value for a 32-bit signed int is 2^31 - 1 = 2147483647.
    # To force a potential overflow in row*M, pick M so that even for a modest row value the product is huge.
    # For example, row=5000 and M=500000 might create a product of 5000*500000 = 2.5e9 which exceeds 2147483647.
    # WARNING: This test allocates a large tensor. Use small sizes if resources are limited.
    try:
        N = 5000
        M = 500000
        # Estimate required memory ~ N*M * 4 bytes, which is about 10GB. If insufficient memory, skip the test.
        total_bytes = N * M * 4
        if total_bytes > torch.cuda.get_device_properties(0).total_memory * 0.5:
            pytest.skip("Test skipped due to GPU memory constraints for simulating index overflow.")

        A = torch.randn(N, dtype=torch.float32, device='cuda')
        B = torch.randn(N, M, dtype=torch.float32, device='cuda')
        module = build_kernel()
        C_kernel = module.forward(A, B)
        
        # Compute expected result using broadcasting (since diag multiplication is row-wise scaling)
        # For each row, multiply the row of B by the corresponding diagonal element.
        C_ref = A.unsqueeze(1) * B

        # If index overflow occurred, the computed C_kernel will differ from C_ref.
        # We expect the maximum absolute difference to exceed a reasonable tolerance.
        assert not torch.allclose(C_kernel, C_ref, atol=1e-5), \
            "Kernel output is (unexpectedly) correct even though index overflow should occur for large sizes."
    except RuntimeError as e:
        # If the GPU allocation fails, skip the test.
        pytest.skip("Skipping index overflow test due to runtime error: " + str(e))

# Issue 3: Test that absence of error checking after kernel launch may hide asynchronous errors.
def test_cuda_error_check():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available, skipping test for CUDA error checking.")
    # We can simulate a situation that causes an error in kernel execution.
    # One way is to intentionally pass a CPU tensor instead of a CUDA tensor.
    N = 1024
    M = 1024

    # Create a CPU tensor for A or B
    A = torch.randn(N, dtype=torch.float32, device='cpu')
    B = torch.randn(N, M, dtype=torch.float32, device='cuda')
    module = build_kernel()
    
    with pytest.raises(RuntimeError):
        # The kernel expects both A and B to be CUDA tensors. This should raise an error.
        module.forward(A, B)
