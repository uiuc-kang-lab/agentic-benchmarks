
import torch
import pytest
from torch.utils.cpp_extension import load
import os

# Build the CUDA kernel module from kernel.cu
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Precision loss for types other than float.
# Test with a double tensor. The output from module.forward will be computed in float and then cast back,
# so we expect the output to differ by more than a strict tolerance from torch.sigmoid computed entirely in double.
def test_precision_loss():
    my_module = build_kernel()
    torch.cuda.synchronize()
    # Create a double tensor with enough elements to fill at least one full block.
    x = torch.randn(1024, device="cuda", dtype=torch.double)
    y = my_module.forward(x)
    y_ref = torch.sigmoid(x)
    # We expect that the difference is significant because of the conversion to float.
    max_diff = (y - y_ref).abs().max().item()
    # Use a very tight tolerance because if there were no precision loss, the outputs would agree.
    assert max_diff > 1e-6, f"Precision loss issue not detected; max diff {max_diff} is too small."

# Issue 2: Hard-coded shared memory size.
# In this test we demonstrate that the kernel only works correctly
# when the block has all threads active. Prepare an input whose length is an exact multiple of the block size.
# (If the block size were made variable, the kernel would need to accommodate different block dimensions.)
def test_hardcoded_shared_memory():
    my_module = build_kernel()
    torch.cuda.synchronize()
    # Create an input with size exactly one block (256 elements).
    # In this case every thread is active and the shared memory buffer (of fixed size 256) is fully used.
    x = torch.randn(256, device="cuda", dtype=torch.float32)
    y = my_module.forward(x)
    y_ref = torch.sigmoid(x)
    assert torch.allclose(y, y_ref, atol=1e-5), "Kernel output incorrect on a full block, which indicates a problem with the hard-coded shared memory configuration."
    # Mark xfail for more general configurations.
    pytest.xfail("Kernel hard-coding of shared memory size makes it incompatible with variable block configurations.")

# Issue 3: Unnecessary use of shared memory.
# While this does not lead to wrong answers under the full-block scenario, it is suboptimal.
# Here we simply run the kernel on a sufficiently large tensor and warn the developer.
def test_unnecessary_shared_memory():
    my_module = build_kernel()
    torch.cuda.synchronize()
    # Use a tensor length that forces multiple blocks (e.g., size that is an exact multiple, to avoid deadlock).
    x = torch.randn(1024, device="cuda", dtype=torch.float32)
    y = my_module.forward(x)
    y_ref = torch.sigmoid(x)
    assert torch.allclose(y, y_ref, atol=1e-5), "Kernel output is incorrect when using large tensors."
    pytest.xfail("Kernel uses unnecessary shared memory despite correct results. This may hurt performance in more complex/general cases.")

# Issue 4: Conditional __syncthreads leading to potential deadlock.
# We create an input whose total number of elements is not a multiple of the block size (256)
# so that in one of the blocks some threads do not satisfy (i < size) and skip __syncthreads.
# This should hang (deadlock) and is detected via a timeout.
@pytest.mark.timeout(5)
def test_deadlock_conditional_syncthreads():
    my_module = build_kernel()
    torch.cuda.synchronize()
    # Create an input with length that is not a multiple of the block size – guaranteed incomplete block.
    # For example, 300 elements: first block (256) all active, second block (44 active, 212 inactive).
    x = torch.randn(300, device="cuda", dtype=torch.float32)
    # This call is expected to hit a deadlock due to conditional __syncthreads.
    with pytest.raises(Exception):
        y = my_module.forward(x)
        torch.cuda.synchronize()
    # If the kernel does not hang, then we xfail this test.
    pytest.xfail("Kernel deadlocks due to conditional __syncthreads when not all block threads are active.")

# Issue 5: Lack of CUDA error checking.
# This test stresses the kernel with a large input.
# Although the kernel may return a result, the lack of error checking means that silent errors could be missed.
def test_missing_error_check():
    my_module = build_kernel()
    torch.cuda.synchronize()
    x = torch.randn(10000, device="cuda", dtype=torch.float32)
    y = my_module.forward(x)
    y_ref = torch.sigmoid(x)
    # Check that the computed result is correct.
    assert torch.allclose(y, y_ref, atol=1e-5), "Kernel output is incorrect."
    pytest.xfail("Kernel host code does not perform error checking after the CUDA launch, risking silent failures.")

