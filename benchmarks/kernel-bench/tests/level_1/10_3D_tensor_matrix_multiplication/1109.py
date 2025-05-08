
import torch
import pytest
from torch.utils.cpp_extension import load
import numpy as np

# Helper to build the CUDA extension
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Incorrect memory transfer.
# Since A and B are already device tensors, using cudaMemcpyAsync from host-to-device is erroneous.
# For a simple input with known values the kernel result (if data is mis-copied) should differ from torch.matmul.
def test_incorrect_memory_transfer():
    my_module = build_kernel()
    # Create small deterministic input (diverse numbers)
    N, M, K, L = 2, 3, 4, 5
    # Use a randn() that is fixed via manual seeding for reproducibility
    torch.manual_seed(42)
    A = torch.randn(N, M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, L, device="cuda", dtype=torch.float32)
    
    # Expected output computed with torch.matmul
    ref = torch.matmul(A, B)
    # Call the CUDA kernel implementation
    out = my_module.forward(A, B)
    torch.cuda.synchronize()
    
    # Because of the wrong memory copy directions, the kernel is likely to produce an incorrect result.
    # We trigger the issue by verifying that the output deviates from the expected result.
    assert not torch.allclose(out, ref, atol=1e-4), \
        f"Test did not trigger incorrect memory transfer: kernel output matches reference output!"

# Issue 2: Race condition because asynchronous copies for A and B use different streams.
# The kernel may be launched before B has been fully copied.
def test_unsynchronized_streams():
    my_module = build_kernel()
    N, M, K, L = 8, 16, 32, 8
    torch.manual_seed(123)
    A = torch.randn(N, M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, L, device="cuda", dtype=torch.float32)
    
    # Compute expected output using torch.matmul
    ref = torch.matmul(A, B)
    # Run the kernel several times to increase the chance that the race condition appears.
    outputs = []
    for _ in range(10):
        out = my_module.forward(A, B)
        torch.cuda.synchronize()
        outputs.append(out)
    
    # If the race condition occurs, some outputs may differ from the correct torch.matmul result.
    deviation_found = any(not torch.allclose(out, ref, atol=1e-4) for out in outputs)
    assert deviation_found, "Test did not trigger the unsynchronized streams issue: All outputs match expected result."

# Issue 3: Inefficient memory allocation and lack of immediate error checking on cudaMalloc.
# We simulate a scenario with large tensors to stress the memory allocation path;
# if cudaMalloc fails, the kernel may print errors or return wrong output.
def test_memory_alloc_failure():
    my_module = build_kernel()
    # Choose sizes that are large enough to stress device memory allocation.
    # Note: This test may be system-dependent. We try to allocate a tensor that is large, but we catch the exception.
    try:
        # Intentionally requesting a huge tensor (may or may not trigger an allocation error on some systems)
        N, M, K, L = 1024, 1024, 1024, 1024
        A = torch.randn(N, M, K, device="cuda", dtype=torch.float32)
        B = torch.randn(K, L, device="cuda", dtype=torch.float32)
        # We do not expect the operation to succeed on every device.
        out = my_module.forward(A, B)
        torch.cuda.synchronize()
        ref = torch.matmul(A, B)
        # With the buggy memory management, the result is likely to be off.
        assert not torch.allclose(out, ref, atol=1e-4), \
            "Test did not trigger memory allocation issues: Kernel output matches torch.matmul unexpectedly."
    except RuntimeError as e:
        # If a CUDA allocation error occurs, then this test has triggered the issue.
        pytest.skip("Memory allocation triggered a RuntimeError as expected: " + str(e))
