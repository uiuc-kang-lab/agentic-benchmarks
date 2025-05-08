
import pytest
import torch
from torch.utils.cpp_extension import load
import math

def build_kernel(extra_cuda_cflags=None):
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=extra_cuda_cflags or ["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1: Incorrect grid-stride step.
#
# Explanation:
#   Under normal use the wrapper’s launch configuration (num_blocks = ceil(n/block_size))
#   guarantees that every element is processed exactly once and the (too‐large) stride
#   is never needed (each thread only does a single iteration). However, if a user were 
#   to override the launch configuration (or if n were so large that additional iterations were required)
#   then a correct grid-stride loop would process multiple iterations.
# 
# In this test we “simulate” a situation where a thread should process more than one element.
# We create an input whose number of elements exceeds the default grid size if we force a smaller grid.
# To do so we recompile the extension with an extra flag that overrides the block size in the kernel’s host code.
# (Imagine that later the kernel is refactored so that grid and block parameters are external.)
#
@pytest.mark.skip(reason="This test simulates a custom launch configuration not used by the default wrapper. "\
                         "It would trigger the stride bug if custom grid/block sizes were allowed.")
def test_incorrect_stride(monkeypatch):
    # Here we simulate a scenario in which the user chooses a block size different from 512.
    # For the test we assume the kernel would have been written to use a custom block size.
    # We force a smaller launch configuration so that each thread must process >1 element.
    N = 1024 * 10   # total elements
    # (simulate a block size of 256 instead of 512 by providing an extra define)
    extra_cflags = ["-O3", "--use_fast_math", "-DBLOCK_SIZE=256"]
    cuda_module = build_kernel(extra_cuda_cflags=extra_cflags)
    # We create input tensors with N elements.
    predictions = torch.rand(N, device="cuda", dtype=torch.float32)
    targets = torch.rand(N, device="cuda", dtype=torch.float32)
    # Compute reference value with PyTorch's smooth_l1_loss (note: reduction="mean")
    loss_ref = torch.nn.functional.smooth_l1_loss(predictions, targets)

    loss_cuda = cuda_module.forward(predictions, targets)
    torch.cuda.synchronize()
    
    # If the grid-stride loop were correct the losses would nearly match.
    # Because our kernel bug causes many elements to be skipped, the returned loss is wrong.
    assert not math.isclose(loss_cuda.item(), loss_ref.item(), rel_tol=1e-3), \
        f"Kernel loss {loss_cuda.item()} unexpectedly matches reference loss {loss_ref.item()} even though the stride bug is present."


# Issue 2: Hard-coded shared memory size.
#
# Explanation:
#   The shared memory array is allocated as __shared__ float shared_sum[512]. If the launch
#   uses a block with more than 512 threads then threads with tid>=512 will write out-of-bounds.
#
# In this test we simulate such a scenario by forcing a larger block size.
@pytest.mark.skip(reason="This test simulates a custom block size >512. The default wrapper always uses 512 threads.")
def test_hard_coded_shared_memory(monkeypatch):
    # For this test we simulate using a block size larger than 512.
    N = 1024
    extra_cflags = ["-O3", "--use_fast_math", "-DBLOCK_SIZE=1024"]
    cuda_module = build_kernel(extra_cuda_cflags=extra_cflags)
    predictions = torch.rand(N, device="cuda", dtype=torch.float32)
    targets = torch.rand(N, device="cuda", dtype=torch.float32)
    # The expected loss computed on CPU with torch.nn.functional.smooth_l1_loss
    loss_ref = torch.nn.functional.smooth_l1_loss(predictions, targets)
    loss_cuda = cuda_module.forward(predictions, targets)
    torch.cuda.synchronize()
    
    # With out-of-bound shared memory writes the result can be corrupted.
    assert not math.isclose(loss_cuda.item(), loss_ref.item(), rel_tol=1e-3), \
        f"Kernel loss {loss_cuda.item()} unexpectedly matches reference loss {loss_ref.item()} even though shared memory overflow is expected."

    
# Issue 3: Lack of type flexibility (only float32 is supported).
#
# Explanation:
#   The kernel unconditionally calls predictions.data_ptr<float>(), so if the input is not float32
#   (for example double) then the result will be computed from erroneously reinterpreted memory.
#
def test_dtype_mismatch():
    cuda_module = build_kernel()
    # Create inputs with double precision instead of float.
    predictions = torch.rand(1024, device="cuda", dtype=torch.float64)
    targets = torch.rand(1024, device="cuda", dtype=torch.float64)
    
    # We expect that using double inputs gives a wrong result (or the kernel might even crash).
    loss_cuda = cuda_module.forward(predictions, targets)
    torch.cuda.synchronize()
    loss_ref = torch.nn.functional.smooth_l1_loss(predictions, targets)
    
    # They should not be nearly the same if the kernel (which only uses float compute)
    assert not torch.allclose(loss_cuda, loss_ref, atol=1e-5), \
        f"Kernel loss {loss_cuda.item()} unexpectedly matches reference loss {loss_ref.item()} even though a dtype mismatch is expected."


if __name__ == "__main__":
    pytest.main([__file__])
