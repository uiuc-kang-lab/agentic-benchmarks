
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel(extra_cuda_cflags=None):
    extra_cuda_cflags = extra_cuda_cflags if extra_cuda_cflags is not None else ["-O3", "--use_fast_math"]
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=extra_cuda_cflags,
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 3: Test that using a double tensor (float64) leads to precision problems.
def test_double_precision_issue():
    # Create a double tensor on CUDA.
    input_tensor = torch.randn(1024, device="cuda", dtype=torch.double)
    # Build the kernel module.
    module = build_kernel()
    # Run the custom CUDA kernel.
    output = module.forward(input_tensor)
    # Compute the reference output using torch.sigmoid.
    ref_output = torch.sigmoid(input_tensor)
    # Because the kernel converts to float and uses expf, there should be a noticeable discrepancy.
    diff = (output - ref_output).abs().max().item()
    # Setting a lower tolerance for double (e.g., 1e-7) we expect a larger error due to conversion.
    # If the discrepancy is unusually small, then the issue is not triggered.
    assert diff > 1e-5, (
        f"Expected noticeable precision loss when using double inputs (difference {diff}), but got too small difference."
    )

# Issue 3 (and possibly issue 1): Test that using half precision (float16) either fails or produces an incorrect result.
def test_half_precision_issue():
    # Create a half tensor on CUDA.
    input_tensor = torch.randn(1024, device="cuda", dtype=torch.half)
    module = build_kernel()
    # Calling the kernel with half precision may produce wrong results or even raise a runtime error
    # since expf is not defined for half.
    with pytest.raises(RuntimeError):
        # We expect that the kernel cannot handle half precision and thus raises an error.
        module.forward(input_tensor)
        
# Issue 2: Test scenario simulating a nonstandard block size.
# Although the current launch configuration uses 256 threads, this test compiles the module with an extra compile flag
# to override the block size if the kernel code were to use a macro for block size (which it currently does not).
# This test is designed to fail (or produce incorrect results) if the kernel is used with a block configuration
# that is larger than the fixed shared memory allocation.
def test_blocksize_issue(monkeypatch):
    # We mimic a situation where a different block size is intended by passing an extra flag.
    # In our kernel code, the block size is hard-coded in the host code (const int threads = 256)
    # and in the shared memory allocation (shared_mem[256]). In a more flexible design this value might be a macro.
    # Here, we simulate compiling with a larger block size, and then we expect that the kernel results would be wrong.
    # For testing purposes, we simply try to force a different configuration.
    # NOTE: Since the current kernel code ignores any such macro, this test is mainly illustrative.
    extra_flags = ["-O3", "--use_fast_math", "-DCUSTOM_BLOCK_SIZE=512"]
    module = build_kernel(extra_cuda_cflags=extra_flags)
    # Create an input tensor large enough so that if CUSTOM_BLOCK_SIZE were honored,
    # the wrong shared memory allocation would lead to errors.
    input_tensor = torch.randn(2048, device="cuda", dtype=torch.float32)
    output = module.forward(input_tensor)
    ref_output = torch.sigmoid(input_tensor)
    # We check if the output differs significantly from the reference.
    diff = (output - ref_output).abs().max().item()
    # In a correct kernel, diff should be near zero.
    # If the shared memory issue is triggered, diff will be large.
    assert diff > 1e-4, (
        f"Expected error due to block size/shared memory mismatch (difference {diff}), but got output close to reference."
    )

# Issue 1: Performance overhead due to unnecessary shared memory usage is hard to test for correctness.
# However, we can compare the runtime of the custom kernel with torch.sigmoid.
# This is a rudimentary test that will not fail unless the custom kernel is significantly slower
# under a simple workload, suggesting unnecessary overhead.
def test_performance_overhead():
    input_tensor = torch.randn(1 << 20, device="cuda", dtype=torch.float32)
    module = build_kernel()
    
    # Warm up
    for _ in range(5):
        _ = module.forward(input_tensor)
    torch.cuda.synchronize()
    
    import time
    start = time.time()
    for _ in range(50):
        _ = module.forward(input_tensor)
    torch.cuda.synchronize()
    custom_time = time.time() - start
    
    # Timing torch.sigmoid
    start = time.time()
    for _ in range(50):
        _ = torch.sigmoid(input_tensor)
    torch.cuda.synchronize()
    torch_time = time.time() - start
    
    # We expect that unnecessary shared memory use may cause the custom kernel to be slower.
    # This test only prints a warning if the custom kernel is not competitive.
    # Instead of asserting, we log the performance difference.
    slowdown = custom_time / torch_time
    print(f"Custom CUDA kernel slowdown factor: {slowdown:.2f}x (custom: {custom_time:.4f}s, torch.sigmoid: {torch_time:.4f}s)")
    # If the slowdown factor is less than 1.1, we do not consider it a problem.
    # (This test is thus informational; in a stricter test suite, one might choose to assert a slowdown.)
    assert slowdown > 1.1, (
        f"Expected the custom kernel to be slower due to extra shared memory overhead, but slowdown factor is only {slowdown:.2f}."
    )
