
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="argmin_kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Non-contiguous input tensor.
def test_non_contiguous_tensor():
    my_module = build_kernel()
    # Create a contiguous tensor then make it non-contiguous by transposing.
    x = torch.randn(4, 8, 16, device="cuda", dtype=torch.float32)
    x_noncontig = x.transpose(0, 1)
    # Compute expected reference using PyTorch's native argmin.
    # Use reduction dim = 1 (in the original layout, after transpose, the reduction axis changes).
    ref = torch.argmin(x_noncontig, dim=1)
    # Call our CUDA kernel (which assumes contiguous layout in the original shape)
    out = my_module.forward(x_noncontig, 1)
    torch.cuda.synchronize()
    # Likely the outputs will be different due to wrong indexing from non-contiguous memory.
    assert not torch.equal(out, ref), "Expected different outputs when using non-contiguous tensor but got the same."

# Issue 2: Reduction dimension of size 0.
def test_zero_reduction_dimension():
    my_module = build_kernel()
    # Create a tensor where the reduction dimension is 0.
    # For example, tensor shape (3, 0, 5) and reduce over dimension 1.
    x = torch.empty(3, 0, 5, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        # This should raise an error or trigger undefined behavior,
        # so we expect our kernel launch to eventually throw an error.
        my_module.forward(x, 1)

# Issue 3: Half precision tensor using operator<.
def test_half_precision_comparison():
    my_module = build_kernel()
    # Create a half precision tensor.
    # Use a tensor with a reduction dimension > 0.
    x = torch.randn(10, 20, device="cuda", dtype=torch.half)
    # PyTorch's built-in argmin should give a correct result.
    ref = torch.argmin(x, dim=1)
    # Use our kernel, which might produce an incorrect result if the half type
    # comparison is not properly implemented.
    out = my_module.forward(x, 1)
    torch.cuda.synchronize()
    # We expect the results to differ if the half-comparison is not working correctly.
    assert not torch.equal(out, ref), "Expected different results for half precision tensor due to improper comparison."

# Issue 4: Incomplete error checking after the kernel launch.
def test_kernel_launch_error_detection():
    my_module = build_kernel()
    # To trigger an asynchronous kernel error, we can deliberately pass an input tensor that is on the CPU.
    x = torch.randn(5, 10, device="cpu", dtype=torch.float32)
    # The kernel should check and throw an error because the input tensor is not on CUDA.
    with pytest.raises(RuntimeError):
        my_module.forward(x, 1)
