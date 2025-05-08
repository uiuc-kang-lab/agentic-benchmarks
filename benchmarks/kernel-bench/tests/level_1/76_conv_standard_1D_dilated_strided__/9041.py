
import threading
import time
import torch
import pytest
from torch.utils.cpp_extension import load

# Utility function to build the extension kernel
def build_kernel():
    module = load(
        name="custom_conv1d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return module

# Issue 1: Test that passing tensors with unsupported dtype (e.g. float64) triggers an error.
def test_unsupported_dtype():
    custom_conv = build_kernel()
    # Create input and weight tensors in double precision
    x = torch.randn(16, 3, 256, device="cuda", dtype=torch.float64)
    weight = torch.randn(64, 3, 3, device="cuda", dtype=torch.float64)
    # (bias remains None)
    # Since the kernel internally casts using data_ptr<float>(),
    # using double tensors should cause incorrect behavior.
    # We test that the tensor type is unsupported.
    with pytest.raises(RuntimeError):
        custom_conv.forward(x, weight, torch.Tensor(), 3, 4)
        
# Issue 2: Test that if cudaMemcpyToSymbol were to fail then no error is reported.
# While it is hard to force a cudaMemcpyToSymbol failure in a unit test,
# we simulate the situation by passing tensors that are on the GPU as normal,
# but we check for inconsistency in the output relative to a CPU version.
def test_constant_memory_copy_error_simulation():
    custom_conv = build_kernel()
    # Use very small weights that should be put into constant memory.
    # We simulate an error by modifying the weights just after the first call.
    # (In a real scenario, a failure in cudaMemcpyToSymbol would have
    # led to corrupted constant memory values.)
    x = torch.randn(1, 2, 20, device="cuda", dtype=torch.float32)
    weight = torch.randn(2, 2, 3, device="cuda", dtype=torch.float32)
    bias = torch.randn(2, device="cuda", dtype=torch.float32)
    # First call – assume correct copy to constant memory.
    output1 = custom_conv.forward(x, weight, bias, stride=1, dilation=1)
    # Simulate corruption by manually changing the weight tensor (this does not affect constant memory)
    # so that if the kernel were mistakenly reading from constant memory it would yield outdated values.
    weight.add_(10.0)
    output2 = custom_conv.forward(x, weight, bias, stride=1, dilation=1)
    # If the cudaMemcpyToSymbol copy were error‐checked correctly, the second call would detect
    # that the constant memory should have been updated. In our simulation, the two outputs will differ.
    # We expect that output1 and output2 do NOT match.
    assert not torch.allclose(output1, output2, atol=1e-5), "Kernel constant memory update did not change output as expected"

# Issue 3: Test for potential race conditions when using constant memory concurrently.
# This test launches two forward calls concurrently (on different threads and streams)
# with different weights that are small enough to use constant memory.
# In a correct design the two calls should be independent. With the constant memory race,
# one call may see the other call's constant memory. We check that at least one result is corrupted.
def test_concurrent_constant_memory_race():
    custom_conv = build_kernel()
    x = torch.randn(4, 3, 50, device="cuda", dtype=torch.float32)
    # Create two different weight sets that both fit in constant memory.
    weight1 = torch.randn(8, 3, 3, device="cuda", dtype=torch.float32)
    weight2 = torch.randn(8, 3, 3, device="cuda", dtype=torch.float32)
    bias1 = torch.randn(8, device="cuda", dtype=torch.float32)
    bias2 = torch.randn(8, device="cuda", dtype=torch.float32)
    
    # Storage for the outputs
    outputs = [None, None]

    # Function to run the kernel call in a thread
    def run_forward(index, weight, bias, stream):
        with torch.cuda.stream(stream):
            out = custom_conv.forward(x, weight, bias, stride=1, dilation=1)
            # Synchronize in this stream before storing the result
            stream.synchronize()
            outputs[index] = out.clone()

    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    # Run both forward calls concurrently in separate threads/streams.
    t1 = threading.Thread(target=run_forward, args=(0, weight1, bias1, stream1))
    t2 = threading.Thread(target=run_forward, args=(1, weight2, bias2, stream2))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    
    # Compute reference outputs using torch.nn.functional.conv1d.
    # Note that our custom kernel does not support any padding so we imitate that.
    import torch.nn.functional as F
    # Reshape weight tensors to F.conv1d expected shape.
    ref1 = F.conv1d(x, weight1, bias1, stride=1, dilation=1)
    ref2 = F.conv1d(x, weight2, bias2, stride=1, dilation=1)
    
    # Because of the unsynchronized use of constant memory,
    # one or both outputs may be incorrect.
    error1 = (outputs[0] - ref1).abs().max().item()
    error2 = (outputs[1] - ref2).abs().max().item()
    # We require that at least one of the outputs is not close to its reference.
    condition = (error1 > 1e-3) or (error2 > 1e-3)
    assert condition, ("Concurrent kernel calls did not reveal a race condition. "
                       "Outputs seem correct even though race conditions are expected in "
                       "a concurrent constant memory update scenario.")
