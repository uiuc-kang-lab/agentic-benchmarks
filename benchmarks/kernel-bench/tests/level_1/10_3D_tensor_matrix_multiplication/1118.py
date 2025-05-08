
import torch
import pytest
import time
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

# Test 1: Trigger the lack of N-dimension parallelism.
# For a relatively large N, the kernel executes the N loop serially.
# Although the final result is correct, the runtime will be significantly slower than torch.matmul.
def test_large_N_parallelism():
    cuda_module = build_kernel()
    # Choose a large N to exaggerate the serial processing in the kernel.
    N, M, K, L = 1024, 256, 128, 64
    A = torch.randn(N, M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, L, device="cuda", dtype=torch.float32)
    
    # Measure time for the custom kernel
    start = time.time()
    C_kernel = cuda_module.forward(A, B)
    torch.cuda.synchronize()
    kernel_time = time.time() - start

    # Measure time for torch.matmul (highly optimized and parallelized)
    start = time.time()
    C_ref = torch.matmul(A, B)
    torch.cuda.synchronize()
    matmul_time = time.time() - start

    # Even though both results should be nearly equal,
    # the custom kernel is expected to be significantly slower due to the lack of N parallelism.
    # We use a loose threshold: the kernel time must be at least 2x slower than torch.matmul.
    # (Note: This test might be sensitive to the underlying hardware.)
    assert torch.allclose(C_kernel, C_ref, atol=1e-3), "Kernel result does not match torch.matmul result."
    assert kernel_time > 2 * matmul_time, f"Kernel did not show the expected slowdown due to serial N loop. Kernel time: {kernel_time}, torch.matmul time: {matmul_time}"

# Test 2: Trigger potential issues with half precision (__ldg usage)
def test_half_precision_issue():
    cuda_module = build_kernel()
    N, M, K, L = 8, 64, 128, 32
    A = torch.randn(N, M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, L, device="cuda", dtype=torch.float16)
    C_kernel = cuda_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    # Use a relaxed tolerance because half precision is low-precision
    if not torch.allclose(C_kernel, C_ref, atol=1e-2):
        raise AssertionError(f"Half precision results differ more than expected! Max diff: {(C_kernel - C_ref).abs().max().item()}")

# Test 3: Trigger error reporting by missing full device synchronization.
# This test ensures that if there’s any asynchronous error in the kernel, it will be caught.
# Here we intentionally pass a non-contiguous tensor, which should trigger a check in the C++ code.
def test_non_contiguous_input():
    cuda_module = build_kernel()
    N, M, K, L = 4, 32, 64, 16
    A = torch.randn(N, M, K, device="cuda", dtype=torch.float32).permute(0, 2, 1)  # Make it non-contiguous
    B = torch.randn(K, L, device="cuda", dtype=torch.float32)
    with pytest.raises(Exception) as excinfo:
        _ = cuda_module.forward(A, B)
    assert "must be contiguous" in str(excinfo.value)
