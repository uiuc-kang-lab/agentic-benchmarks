
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Helper function to compile the extension.
def build_kernel(extra_cuda_flags=None):
    extra_cuda_flags = extra_cuda_flags or ["-O3", "--use_fast_math"]
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=[],
        extra_cuda_cflags_nv=extra_cuda_flags,
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Misaligned memory accesses due to vectorized loads.
#    We trigger the issue by creating mis-aligned tensors. One trick is to create a tensor
#    with extra elements and then take a narrow view starting from offset 1 along the feature dim.
def test_misaligned_memory():
    batch_size = 4
    feat_size = 17  # arbitrary size not a multiple of 4
    # Create a tensor with an extra element along the feature dimension.
    anchor_large = torch.randn(batch_size, feat_size + 1, device="cuda", dtype=torch.float32)
    positive_large = torch.randn(batch_size, feat_size + 1, device="cuda", dtype=torch.float32)
    negative_large = torch.randn(batch_size, feat_size + 1, device="cuda", dtype=torch.float32)
    # Create misaligned views by slicing away the first feature column.
    anchor = anchor_large.narrow(1, 1, feat_size)
    positive = positive_large.narrow(1, 1, feat_size)
    negative = negative_large.narrow(1, 1, feat_size)
    
    mod = build_kernel()
    # Expect that the misalignment causes an illegal memory access or wrong output.
    with pytest.raises(RuntimeError):
        loss = mod.forward(anchor, positive, negative)
        # Force synchronization to catch asynchronous CUDA errors.
        torch.cuda.synchronize()

# Issue 2: Assumption on blockDim and the fixed shared memory array size.
#    We trigger the issue by compiling the kernel with an overridden thread count value
#    that exceeds 1024. This simulates a scenario in which the number of warps per block exceeds
#    the size of the statically allocated shared arrays.
def test_excessive_thread_block():
    # Recompile the module with an extra CUDA flag that defines THREADS to a value > 1024.
    # Note: This requires that the kernel.cu file is written to allow overriding the thread count
    # via a compile time macro (e.g. using "#ifndef THREADS\n#define THREADS 256\n#endif" and then 
    # "int threads = THREADS;" in place of the literal 256).
    extra_flags = ["-O3", "--use_fast_math", "-DTHREADS=1056"]
    mod = build_kernel(extra_cuda_flags=extra_flags)
    
    batch_size = 4
    feat_size = 128  # arbitrary feature size
    anchor = torch.randn(batch_size, feat_size, device="cuda", dtype=torch.float32)
    positive = torch.randn(batch_size, feat_size, device="cuda", dtype=torch.float32)
    negative = torch.randn(batch_size, feat_size, device="cuda", dtype=torch.float32)
    
    # We expect that using an excessive thread block will lead to shared memory corruption,
    # which in turn should produce an incorrect (non-finite) result.
    loss = mod.forward(anchor, positive, negative)
    torch.cuda.synchronize()
    # If shared memory corruption occurs, the kernel output might be NaN or inf.
    assert not torch.isfinite(loss), "Expected non-finite loss due to shared memory overrun, but got a finite value."

# Issue 3: Data type assumption.
#    The kernel only supports inputs of type float (float32). Here we pass double tensors to trigger the issue.
def test_wrong_dtype():
    batch_size = 4
    feat_size = 128
    anchor = torch.randn(batch_size, feat_size, device="cuda", dtype=torch.float64)
    positive = torch.randn(batch_size, feat_size, device="cuda", dtype=torch.float64)
    negative = torch.randn(batch_size, feat_size, device="cuda", dtype=torch.float64)
    
    mod = build_kernel()
    # The kernel performs pointer reinterpretation assuming float data.
    # This should lead to an incorrect result or a runtime error.
    with pytest.raises(RuntimeError):
        loss = mod.forward(anchor, positive, negative)
        torch.cuda.synchronize()
