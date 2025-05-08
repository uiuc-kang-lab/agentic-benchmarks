
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    # Build and load the CUDA extension from kernel.cu
    cuda_module = load(
        name="depthwise_conv2d_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Test that the kernel fails when non-float32 tensors are provided.
# The kernel code uses data_ptr<float>(), so passing a tensor of type double should trigger an error.
def test_dtype_error():
    cuda_module = build_kernel()
    batch_size = 1
    in_channels = 1
    input_h = 8
    input_w = 8
    kernel_size = 3

    # Create double tensors intentionally.
    input_tensor = torch.randn(batch_size, in_channels, input_h, input_w, dtype=torch.float64, device="cuda")
    # Weight shape: [in_channels, channel_multiplier, kernel_size, kernel_size].
    # For a depthwise conv with multiplier 1 the shape is (in_channels, 1, kernel_size, kernel_size).
    weight_tensor = torch.randn(in_channels, 1, kernel_size, kernel_size, dtype=torch.float64, device="cuda")
    bias_tensor = torch.randn(in_channels, dtype=torch.float64, device="cuda")

    # Expect a runtime error due to improper type (float64 instead of float32).
    with pytest.raises(RuntimeError):
        # Here stride and padding are arbitrary valid integers.
        _ = cuda_module.forward(input_tensor, weight_tensor, bias_tensor, 1, 0)


# Issue 2: Test that the kernel fails when provided non-contiguous inputs.
# The kernel expects contiguous tensors; non-contiguous inputs should trigger the TORCH_CHECK.
def test_non_contiguous_error():
    cuda_module = build_kernel()
    batch_size = 1
    in_channels = 1
    input_h = 8
    input_w = 8
    kernel_size = 3

    # Create a contiguous tensor then make it non-contiguous (e.g. by transposing spatial dims).
    input_tensor = torch.randn(batch_size, in_channels, input_h, input_w, device="cuda", dtype=torch.float32)
    non_contig_input = input_tensor.transpose(2, 3)  # Now non-contiguous.

    weight_tensor = torch.randn(in_channels, 1, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias_tensor = torch.randn(in_channels, device="cuda", dtype=torch.float32)

    with pytest.raises(RuntimeError):
        _ = cuda_module.forward(non_contig_input, weight_tensor, bias_tensor, 1, 0)


# Issue 3: Test for kernel launch problems.
# Although the unused shared memory itself may not trigger a runtime error,
# we can simulate a problematic launch by providing invalid kernel parameters.
# For example, if stride is set to 0 then the output dimensions become invalid (division by zero in output size calculation)
# and the kernel launch will likely produce an error.
def test_invalid_stride():
    cuda_module = build_kernel()
    batch_size = 1
    in_channels = 1
    input_h = 8
    input_w = 8
    kernel_size = 3

    input_tensor = torch.randn(batch_size, in_channels, input_h, input_w, device="cuda", dtype=torch.float32)
    weight_tensor = torch.randn(in_channels, 1, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias_tensor = torch.randn(in_channels, device="cuda", dtype=torch.float32)

    # Use an invalid stride (e.g., 0) to force an error in the computation of output dimensions.
    with pytest.raises(Exception):
        _ = cuda_module.forward(input_tensor, weight_tensor, bias_tensor, 0, 0)
