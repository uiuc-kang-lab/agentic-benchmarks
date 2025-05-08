
import pytest
import torch
from torch.utils.cpp_extension import load

# Utility function: builds the CUDA extension from the kernel file "kernel.cu".
def build_kernel():
    cuda_module = load(
        name="conv_transpose1d_kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Test that using a non-float32 tensor (e.g. float64) triggers a failure.
def test_non_float32_dtype():
    cuda_module = build_kernel()
    # Create input tensors in float64, which the kernel does not support.
    batch_size = 2
    in_channels = 2
    out_channels = 3
    length = 10
    kernel_size = 3
    stride = 1
    padding = 0
    dilation = 1

    # Inputs are double precision.
    x = torch.randn(batch_size, in_channels, length, device="cuda", dtype=torch.float64)
    weight = torch.randn(in_channels, out_channels, kernel_size, device="cuda", dtype=torch.float64)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float64)

    # Expect that the extension, which casts the tensor data pointers to float*, misinterprets the memory.
    # We run the forward operation and then check whether the returned output is nonsensical.
    # Since the operation is not supported, we expect a significant difference from the reference (by re-casting),
    # or possibly a runtime error.
    with pytest.raises(RuntimeError):
        # Even if no explicit type check is performed in the kernel, using double
        # data will yield wrong computations that we can check.
        out = cuda_module.forward(x, weight, bias, stride, padding, dilation)
        # Force a synchronization to catch asynchronous errors.
        torch.cuda.synchronize()

# Issue 2: Test that providing a bias tensor with incorrect shape exposes the lack of proper shape validation.
def test_incorrect_bias_shape():
    cuda_module = build_kernel()
    # Proper dimensions for conv_transpose1d parameters:
    batch_size = 2
    in_channels = 3
    out_channels = 4
    length = 12
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1

    # Create valid input and weight tensors.
    x = torch.randn(batch_size, in_channels, length, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels, kernel_size, device="cuda", dtype=torch.float32)
    # Provide a bias tensor with incorrect shape (should be [out_channels] but here is [1]).
    bias = torch.randn(1, device="cuda", dtype=torch.float32)

    # With the bias shape wrong, accessing bias[c_out] may be out-of-bounds.
    with pytest.raises(RuntimeError):
        out = cuda_module.forward(x, weight, bias, stride, padding, dilation)
        torch.cuda.synchronize()
