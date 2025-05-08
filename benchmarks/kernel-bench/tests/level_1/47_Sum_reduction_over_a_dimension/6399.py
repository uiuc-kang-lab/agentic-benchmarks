
import torch
import pytest
from torch.utils.cpp_extension import load
import time

def build_kernel():
    cuda_module = load(
        name="test_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Concurrent kernel launches that may trigger race conditions on __constant__ memory.
# Launch two reduction operations concurrently on different streams with different reduction sizes.
def test_concurrent_kernel_launch():
    cuda_module = build_kernel()
    # Create two inputs with different shapes and reduction dimensions.
    # The first input reduces over dim=1 (size 128) and the second reduces over dim=1 (size 64).
    input1 = torch.randn(4, 128, 32, device="cuda", dtype=torch.float32)
    input2 = torch.randn(4, 64, 32, device="cuda", dtype=torch.float32)
    
    # Use two separate streams to force concurrent kernel launches.
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    
    # Launch the kernel asynchronously on two streams WITHOUT an explicit synchronization in between.
    with torch.cuda.stream(stream1):
        out1 = cuda_module.forward(input1, 1)  # reduce over dimension 1
    with torch.cuda.stream(stream2):
        out2 = cuda_module.forward(input2, 1)  # reduce over dimension 1
    
    # Synchronize both streams
    stream1.synchronize()
    stream2.synchronize()
    
    # Expected outputs computed with PyTorch's sum.
    expected1 = torch.sum(input1, dim=1, keepdim=True)
    expected2 = torch.sum(input2, dim=1, keepdim=True)

    # These assertions might fail if constant memory values got overwritten between launches.
    assert torch.allclose(out1, expected1, atol=1e-5), "Concurrent kernel launch output for input1 is incorrect."
    assert torch.allclose(out2, expected2, atol=1e-5), "Concurrent kernel launch output for input2 is incorrect."

# Test 2: Non-contiguous input tensor.
# Create an input tensor that is non-contiguous due to a transpose operation and check if the result is correct.
def test_non_contiguous_input():
    cuda_module = build_kernel()
    # Create an input, then perform a transpose to make it non-contiguous.
    input_orig = torch.randn(8, 16, 32, device="cuda", dtype=torch.float32)
    input_non_contig = input_orig.transpose(0, 1)  # now shape is (16, 8, 32); non-contiguous layout
    # For reduction, pick the middle dimension (dim=1).
    # Compute expected output, but note that torch.sum works correctly even on non-contiguous tensors.
    expected = torch.sum(input_non_contig, dim=1, keepdim=True)
    
    # The kernel does not honor strides so its output may be wrong.
    out = cuda_module.forward(input_non_contig, 1)
    
    # This assertion is expected to fail if the non-contiguous input is not handled properly.
    assert torch.allclose(out, expected, atol=1e-5), "Kernel output for non-contiguous input is incorrect."

# Test 3: Passing a CPU tensor (should trigger an error).
def test_cpu_input():
    cuda_module = build_kernel()
    input_cpu = torch.randn(4, 32, 16, device="cpu", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        # Expecting the CUDA kernel call to throw an error because the input is not on CUDA.
        cuda_module.forward(input_cpu, 1)
        
if __name__ == "__main__":
    pytest.main([__file__])
