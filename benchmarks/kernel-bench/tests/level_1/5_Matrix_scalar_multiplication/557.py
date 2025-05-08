
import torch
import pytest
from torch.utils.cpp_extension import load
import threading
import time

def build_kernel():
    # This builds the CUDA extension from the provided kernel.cu file.
    cuda_module = load(
        name="my_cuda_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Test 1: Non-contiguous / misaligned input
# Create a non-contiguous tensor by transposing a contiguous tensor.
def test_non_contiguous_input():
    my_module = build_kernel()
    # Create a contiguous tensor and then transpose it to make it non-contiguous.
    M, N = 128, 256
    A = torch.randn(M, N, device="cuda", dtype=torch.float32).t()  # non-contiguous now
    s = 2.0
    # The expected behaviour is that the kernel fallback to simple kernel works,
    # but if the non-contiguity causes an issue, the result will be wrong.
    C = my_module.forward(A, s)
    torch.cuda.synchronize()
    C_ref = A * s
    assert torch.allclose(C, C_ref, atol=1e-5), "Non-contiguous input did not produce expected results."

# Test 2: Concurrent kernel launches with different scalar values to trigger race on constant memory.
def test_concurrent_scalar_race():
    my_module = build_kernel()
    M, N = 512, 512
    # Two different scalar values for concurrent invocations.
    s1 = 3.0
    s2 = 4.0
    A1 = torch.randn(M, N, device="cuda", dtype=torch.float32)
    A2 = torch.randn(M, N, device="cuda", dtype=torch.float32)
    results = {}

    def run_kernel(name, tensor, scalar):
        # Using separate streams to run kernels concurrently.
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            out = my_module.forward(tensor, scalar)
            stream.synchronize()
        results[name] = out

    thread1 = threading.Thread(target=run_kernel, args=("r1", A1, s1))
    thread2 = threading.Thread(target=run_kernel, args=("r2", A2, s2))
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()

    C1_ref = A1 * s1
    C2_ref = A2 * s2
    # If the constant memory race happens, one or both outputs will be incorrect.
    assert torch.allclose(results["r1"], C1_ref, atol=1e-5), "Concurrent execution: Kernel result with scalar s1 is incorrect."
    assert torch.allclose(results["r2"], C2_ref, atol=1e-5), "Concurrent execution: Kernel result with scalar s2 is incorrect."

# Test 3: Incorrect data type for input tensor (non-float32)
def test_wrong_dtype():
    my_module = build_kernel()
    M, N = 64, 64
    A = torch.randn(M, N, device="cuda", dtype=torch.float64)  # double instead of float
    s = 2.0
    with pytest.raises(RuntimeError, match="Input tensor A must be of type float."):
        my_module.forward(A, s)

# Test 4: Launching with a CPU tensor should raise an error.
def test_cpu_tensor_input():
    my_module = build_kernel()
    M, N = 64, 64
    A = torch.randn(M, N, device="cpu", dtype=torch.float32)
    s = 2.0
    with pytest.raises(RuntimeError, match="Input tensor A must be a CUDA tensor."):
        my_module.forward(A, s)
