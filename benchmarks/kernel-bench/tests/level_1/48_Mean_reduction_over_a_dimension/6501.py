
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the CUDA extension module "test_module" from kernel.cu
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test case 1: Trigger issue with non-contiguous tensor.
def test_non_contiguous():
    # Create a contiguous tensor and then make it non-contiguous by a transpose operation.
    # For example, create a tensor of shape [4, 6, 8] and transpose the first two dims.
    x = torch.randn(4, 6, 8, device="cuda", dtype=torch.float32)
    x_non_contig = x.transpose(0, 1)  # now non-contiguous
    # Compute mean reduction along dim=1 using the custom kernel extension.
    my_module = build_kernel()
    # The kernel expects an input tensor and a reduction dim; we pass dim=1.
    y_kernel = my_module.forward(x_non_contig, 1)
    # Reference using PyTorch's mean (which correctly handles non-contiguous tensors)
    y_ref = torch.mean(x_non_contig, dim=1)
    # Since our kernel assumes contiguous layout, the result may be wrong.
    assert not torch.allclose(y_kernel, y_ref, atol=1e-5), "Kernel unexpectedly worked with a non-contiguous tensor."

# Test case 2: Trigger issue with blockDim.x assumption.
# Although the kernel launch is hard-coded to use 256 threads (a multiple of 32), 
# we can trigger potential issues by providing a tensor where the reduction dimension is small,
# so that many threads perform no work. This may reveal problems if the kernel relies on the
# assumption that every warp is fully active.
def test_small_reduction_dimension():
    # Create a tensor with a very small reduction dimension (size 3) and non-trivial outer/inner sizes.
    x = torch.randn(5, 3, 7, device="cuda", dtype=torch.float32)
    my_module = build_kernel()
    y_kernel = my_module.forward(x, 1)
    y_ref = torch.mean(x, dim=1)
    # If the kernel does not properly deal with block size not matching the reduction dimension,
    # the results may be incorrect.
    assert not torch.allclose(y_kernel, y_ref, atol=1e-5), "Kernel unexpectedly worked with a small reduction dimension."

# Test case 3: Trigger issue with unsupported tensor types.
def test_unsupported_type():
    # Create an integer tensor; torch.mean may be defined for such tensors in Python
    # (by promoting the result), but our kernel dispatch covers only floating types.
    x = torch.randint(0, 10, (8, 16, 32), device="cuda", dtype=torch.int32)
    my_module = build_kernel()
    with pytest.raises(RuntimeError):
        # Expect a runtime error due to unsupported type.
        my_module.forward(x, 1)

# Test case 4: Trigger issue with empty reduction dimension.
def test_empty_reduction_dimension():
    # Create a tensor where the dimension to be reduced is empty.
    # e.g., shape = [7, 0, 5]; reduction along dim=1 yields an empty tensor.
    x = torch.randn(7, 0, 5, device="cuda", dtype=torch.float32)
    my_module = build_kernel()
    # The kernel does not check for size==0 in the reduction dimension and might crash or yield NaNs.
    # We catch any exception and also check if the result is not a valid tensor.
    with pytest.raises(Exception):
        # This should raise an exception or lead to undefined behavior.
        _ = my_module.forward(x, 1)
