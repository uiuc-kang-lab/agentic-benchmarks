
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Helper function to rebuild the CUDA module
def build_kernel(extra_cuda_flags=None):
    extra_cuda_flags = extra_cuda_flags or ["-O3", "--use_fast_math"]
    cuda_module = load(
        name="test_module",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3"],
        extra_cuda_flags=extra_cuda_flags,
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_dtype_support_fail():
    # Issue 1: Kernel only supports float32.
    # Create inputs using float16 to trigger an error.
    batch_size = 2
    in_channels = 4
    out_channels = 6
    kernel_size = 3
    input_length = 10
    stride = 1
    padding = 0
    dilation = 1

    x = torch.randn(batch_size, in_channels, input_length, device="cuda", dtype=torch.float16)
    # Weight expected shape: (in_channels, out_channels, kernel_size)
    weight = torch.randn(in_channels, out_channels, kernel_size, device="cuda", dtype=torch.float16)
    bias = None

    mod = build_kernel()
    with pytest.raises(RuntimeError):
        # The kernel will try to read the data as float, so an error is expected.
        out = mod.forward(x, weight, bias, stride, padding, dilation)
        torch.cuda.synchronize()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_grouped_convolution_mismatch():
    # Issue 2: Kernel does not support groups.
    # In grouped convolution, the weight layout is (in_channels, out_channels_per_group, kernel_size)
    # But our kernel uses in_channels * out_channels * kernel_size.
    batch_size = 2
    in_channels = 4
    groups = 2
    # For grouped conv, out_channels should be groups * (desired channels per group)
    out_channels = 6  # desired overall out_channels
    kernel_size = 3
    input_length = 10
    stride = 1
    padding = 0
    dilation = 1

    # Create input and weights for a grouped convolution.
    x = torch.randn(batch_size, in_channels, input_length, device="cuda", dtype=torch.float32)
    # Typical grouped weight shape: (in_channels, out_channels_per_group, kernel_size)
    out_channels_per_group = out_channels // groups
    weight = torch.randn(in_channels, out_channels_per_group, kernel_size, device="cuda", dtype=torch.float32)
    
    # Call the kernel assuming the flat layout but supply grouped weights.
    mod = build_kernel()
    # The kernel code will use weight.size(1) as out_channels; 
    # so output shape will be wrong relative to a grouped expected behavior.
    out = mod.forward(x, weight, None, stride, padding, dilation)
    # Here we expect the output shape to be different from the one of a grouped conv.
    # If the kernel were used in a grouped situation, the shape would be inconsistent.
    expected_out_channels = weight.size(1)  # which is out_channels_per_group instead of out_channels.
    expected_output_length = (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    assert out.shape[1] == expected_out_channels, (
        f"Output channel mismatch: kernel ignored group partitioning! "
        f"Expected {expected_out_channels}, got {out.shape[1]}"
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_shared_memory_overuse():
    # Issue 3: The kernel does not check for shared memory size.
    # We simulate a situation that likely exceeds shared memory limits.
    # While exact limits depend on the GPU, using a very large weight size should trigger an error.
    batch_size = 1
    in_channels = 512
    out_channels = 512
    kernel_size = 9
    input_length = 20
    stride = 1
    padding = 0
    dilation = 1

    x = torch.randn(batch_size, in_channels, input_length, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    mod = build_kernel()
    # Attempting the operation is expected to fail because shared memory required
    # (in_channels*out_channels*kernel_size*sizeof(float)) is huge.
    with pytest.raises(Exception):
        out = mod.forward(x, weight, bias, stride, padding, dilation)
        torch.cuda.synchronize()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_missing_launch_error_check():
    # Issue 4: The kernel launch does not check errors.
    # Provide configuration that is invalid (e.g. negative stride) so that the kernel logic fails.
    batch_size = 2
    in_channels = 4
    out_channels = 5
    kernel_size = 3
    input_length = 12
    stride = -1  # Invalid stride value to force potential miscomputation.
    padding = 0
    dilation = 1

    x = torch.randn(batch_size, in_channels, input_length, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    mod = build_kernel()
    # We expect the kernel to produce incorrect results or raise an error upon synchronization.
    out = mod.forward(x, weight, bias, stride, padding, dilation)
    with pytest.raises(RuntimeError):
        # Try to synchronize to catch any launch errors
        torch.cuda.synchronize()
