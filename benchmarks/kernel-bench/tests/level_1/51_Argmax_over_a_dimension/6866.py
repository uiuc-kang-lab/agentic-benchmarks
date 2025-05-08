
import torch
import pytest
from torch.utils.cpp_extension import load
import threading

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Trigger non-float32 input issue.
def test_non_float32_input():
    my_module = build_kernel()
    # Create input tensor of type double to trigger the type check.
    x = torch.randn(16, 256, 256, device="cuda", dtype=torch.double)
    # Expecting a RuntimeError from TORCH_CHECK inside the C++ function.
    with pytest.raises(RuntimeError):
        my_module.forward(x, 1)  # Passing dim=1 as usual

# Test 2: Trigger constant memory race condition.
# We simulate concurrency by launching kernels in two different streams with different dim sizes.
def test_concurrent_kernel_launches():
    my_module = build_kernel()
    
    # Prepare two inputs with different reduction dimensions.
    # First tensor: perform argmax along dim=1 (dimSize = 256)
    x1 = torch.randn(8, 256, 32, device="cuda", dtype=torch.float32)
    # Second tensor: perform argmax along dim=2 (dimSize = 32)
    x2 = torch.randn(8, 32, 256, device="cuda", dtype=torch.float32)
    
    # Function to run forward on provided input and dimension.
    def run_argmax(x, dim, out_list, idx, stream):
        with torch.cuda.stream(stream):
            out = my_module.forward(x, dim)
            # Force synchronization in this stream.
            torch.cuda.synchronize()
            out_list[idx] = out.clone()
    
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    
    outputs = [None, None]
    # Launch both kernel calls concurrently in separate threads.
    thread1 = threading.Thread(target=run_argmax, args=(x1, 1, outputs, 0, stream1))
    thread2 = threading.Thread(target=run_argmax, args=(x2, 2, outputs, 1, stream2))
    
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()
    
    # Compute reference results using PyTorch's argmax.
    ref1 = torch.argmax(x1, dim=1)
    ref2 = torch.argmax(x2, dim=2)
    
    # The race condition on c_dimSize may cause one of the outputs
    # to be computed with the wrong dimension size.
    # We check if at least one output does not match its reference.
    err1 = not torch.equal(outputs[0], ref1)
    err2 = not torch.equal(outputs[1], ref2)
    assert err1 or err2, "Concurrent kernel launches did not trigger the constant memory race condition as expected."

# Test 3: Trigger missing error checking on cudaMemcpyToSymbol.
# Although it is not trivial to force a failure of cudaMemcpyToSymbol,
# we simulate a scenario where we call the kernel with a large dimension that might cause issues.
def test_large_dim_size():
    my_module = build_kernel()
    # Create a tensor with a very large size along the reduction dimension.
    # This tests whether c_dimSize update and subsequent memory accesses are safe.
    x = torch.randn(4, 1024, 8, device="cuda", dtype=torch.float32)
    # We don't necessarily expect a wrong result, but we want to verify that the kernel doesn't silently fail.
    output = my_module.forward(x, 1)
    ref = torch.argmax(x, dim=1)
    # Allow for potential numerical differences (but in argmax, they must be exactly same).
    assert torch.equal(output, ref), "Kernel result for large dim size does not match PyTorch's argmax."

if __name__ == '__main__':
    pytest.main([__file__])
