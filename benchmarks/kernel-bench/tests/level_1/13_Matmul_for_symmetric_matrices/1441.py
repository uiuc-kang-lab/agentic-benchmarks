
import torch
import pytest
from torch.utils.cpp_extension import load
import threading
import time

def build_kernel():
    kernel_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return kernel_module

# Issue 2: Test that using a non-float32 tensor (e.g., double) triggers an error or produces a wrong result.
def test_input_tensor_type():
    # Create double precision symmetric matrices
    N = 256
    A = torch.randn(N, N, dtype=torch.double, device='cuda')
    A = (A + A.t()) / 2
    B = torch.randn(N, N, dtype=torch.double, device='cuda')
    B = (B + B.t()) / 2

    my_module = build_kernel()
    # We expect that passing double tensors will lead to a failure, either throwing an error or producing an incorrect result.
    with pytest.raises(RuntimeError):
        # Likely, the kernel will try to interpret the bits as float, causing a runtime issue.
        _ = my_module.forward(A, B)
        
# Issue 1: Test concurrent kernel launches with different matrix sizes to expose the unsafe usage of __constant__ memory.
def test_concurrent_constant_memory():
    my_module = build_kernel()
    
    # Function that runs the kernel forward pass in a dedicated CUDA stream
    def run_forward(N, results, index, delay=0):
        # Create symmetric matrices of size N x N
        A = torch.randn(N, N, dtype=torch.float32, device='cuda')
        A = (A + A.t()) / 2
        B = torch.randn(N, N, dtype=torch.float32, device='cuda')
        B = (B + B.t()) / 2

        # Use a dedicated CUDA stream
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            # Optionally delay the launch to interleave kernels from different streams
            if delay:
                time.sleep(delay)
            C = my_module.forward(A, B)
        # Wait for this stream to complete before storing result
        torch.cuda.synchronize()
        results[index] = (N, C, torch.matmul(A, B))

    # We will launch two threads concurrently in different streams with different matrix sizes.
    results = {}
    t1 = threading.Thread(target=run_forward, args=(256, results, 1, 0))
    t2 = threading.Thread(target=run_forward, args=(512, results, 2, 0.01))
    
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    
    # Compare the computed outputs with the reference using torch.matmul.
    N1, C1, C1_ref = results[1]
    N2, C2, C2_ref = results[2]
    
    # Since the kernel uses global constant memory, interleaving kernel launches may cause the parameters to be overwritten,
    # leading to errors in one (or both) of the results.
    # We expect at least one of these comparisons to fail.
    error1 = (C1 - C1_ref).abs().max().item()
    error2 = (C2 - C2_ref).abs().max().item()
    
    # The tolerance here is loose; if the constant memory issue occurs then the max error will be huge.
    # We assert that at least one of the results is not close to the expected result.
    assert error1 > 1e-3 or error2 > 1e-3, (
        f"Kernel outputs appear correct despite concurrent launches. "
        f"Error1: {error1}, Error2: {error2}. This indicates the __constant__ memory usage is not interfering as expected."
    )
    
if __name__ == '__main__':
    pytest.main([__file__])
