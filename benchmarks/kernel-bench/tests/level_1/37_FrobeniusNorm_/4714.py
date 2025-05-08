
import os
import math
import pytest
import torch
from torch.utils.cpp_extension import load

# A helper function to build the CUDA extension.
# For the shared memory issue test we define a macro FORCE_BAD_BLOCK_DIM=1 (which the kernel code can use to force a high block count).
def build_kernel(extra_cuda_cflags=None):
    extra_cuda_cflags = extra_cuda_cflags or ["-O3", "--use_fast_math"]
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=extra_cuda_cflags,
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Non–float32 input tensor (Issue 4: although the TORCH_CHECK protects the type, this test shows that non–float32 inputs are rejected)
def test_input_tensor_type():
    my_module = build_kernel()
    # Create an input tensor with type float64 on GPU.
    x = torch.randn(16, 64, 256, 256, dtype=torch.float64, device="cuda")
    with pytest.raises(RuntimeError, match="Input must be float32"):
        # Should trigger the TORCH_CHECK in forward().
        _ = my_module.forward(x)

# Test 2: Non–contiguous input tensor trigger check (even if not an issue in many kernels, our code requires contiguous input)
def test_non_contiguous_tensor():
    my_module = build_kernel()
    x = torch.randn(16, 64, 256, 256, dtype=torch.float32, device="cuda")
    x_noncontig = x.transpose(1, 2)  # produce a non-contiguous tensor
    with pytest.raises(RuntimeError, match="Input tensor must be contiguous"):
        _ = my_module.forward(x_noncontig)

# Test 3: Zero norm input (Issue 2). An input where all elements are zero produces a zero norm so the normalization will perform a division by zero.
def test_zero_norm():
    my_module = build_kernel()
    x = torch.zeros(16, 64, 256, 256, dtype=torch.float32, device="cuda")
    # Here, the Frobenius norm is zero and division by zero will occur.
    output = my_module.forward(x)
    # The output is undefined (likely NaN) so we check that it contains NaNs.
    assert torch.isnan(output).any(), "Output should contain NaNs when input norm is zero"

# Test 4: Shared memory block size issue.
# In real usage the kernel assumes 256 threads per block but we simulate a case where we launch the kernel with a larger block size.
# For this test we assume that kernel.cu is modified to check for a macro (e.g. BAD_BLOCK_DIM) and if defined,
# it sets the number of threads (or relies on a macro THREADS_PER_BLOCK) that is larger than 256.
#
# For demonstration, we pass a macro definition that forces THREADS_PER_BLOCK to 512.
# (In a production version, you would parameterize your launch configuration and shared memory allocation.)
def test_shared_memory_issue():
    # Build a version of the extension that forces a larger block size.
    my_module = build_kernel(extra_cuda_cflags=["-O3", "--use_fast_math", "-DBAD_BLOCK_DIM", "-DTHREADS_PER_BLOCK=512"])
    # Create a small tensor so that even if the kernel reads out-of–bounds in shared memory the artifact is likely to lead to an error.
    x = torch.randn(16, 64, 16, 16, dtype=torch.float32, device="cuda")
    # Depending on the system this may cause a CUDA error (like illegal memory access).
    # We catch the RuntimeError; if no error is thrown then the test fails because the bug is not triggered.
    with pytest.raises(RuntimeError):
        _ = my_module.forward(x)
