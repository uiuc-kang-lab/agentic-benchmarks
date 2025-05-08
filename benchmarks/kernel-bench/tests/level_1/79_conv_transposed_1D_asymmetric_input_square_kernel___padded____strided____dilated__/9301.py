
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    # Compile the CUDA extension from kernel.cu
    cuda_module = load(
        name="test_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Test 1: Passing non float32 (double) data should trigger an error or produce a mismatch.
def test_dtype_error():
    cuda_module = build_kernel()
    batch_size = 2
    in_channels = 4
    out_channels = 3
    kernel_size = 3
    input_length = 10
    stride = 2
    padding = 1
    dilation = 1

    # Create double precision tensors on CUDA. The kernel expects float32.
    x = torch.randn(batch_size, in_channels, input_length, dtype=torch.float64, device="cuda")
    # Weight of shape: (in_channels, out_channels, kernel_size)
    weight = torch.randn(in_channels, out_channels, kernel_size, dtype=torch.float64, device="cuda")
    bias = torch.randn(out_channels, dtype=torch.float64, device="cuda")
    
    with pytest.raises(RuntimeError):
        # The following call will eventually use data_ptr<float>() on a double tensor.
        cuda_module.forward(x, weight, bias, stride, padding, dilation)
    
    
# Test 2: Grouped convolution is not supported.
def test_grouped_convolution():
    cuda_module = build_kernel()
    batch_size = 2
    in_channels = 4
    out_channels = 6
    kernel_size = 3
    input_length = 10
    stride = 2
    padding = 1
    dilation = 1
    groups = 2  # Suppose we want groups=2; then each group should have in_channels/groups.

    # For a grouped conv, expected weight shape should be (in_channels, out_channels/groups, kernel_size)
    # However, the kernel expects weight shape (in_channels, out_channels, kernel_size).
    # We simulate providing a weight tensor with a grouped shape.
    group_out_channels = out_channels // groups
    x = torch.randn(batch_size, in_channels, input_length, dtype=torch.float32, device="cuda")
    # This weight shape intentionally mismatches what the kernel expects.
    weight = torch.randn(in_channels, group_out_channels, kernel_size, dtype=torch.float32, device="cuda")
    bias = torch.randn(group_out_channels, dtype=torch.float32, device="cuda")
    
    with pytest.raises(RuntimeError):
        # The kernel checks that weight.size(0) equals x.size(1). This passes,
        # but then later the accumulation over output channels will be computed wrong.
        # Our test expects that a mismatch in layout/size will eventually raise an error.
        cuda_module.forward(x, weight, bias, stride, padding, dilation)
    
    
# Test 3: Passing tensors with wrong dimensions should be caught.
def test_tensor_dimension_error():
    cuda_module = build_kernel()
    batch_size = 2
    in_channels = 4
    out_channels = 3
    kernel_size = 3
    input_length = 10
    stride = 2
    padding = 1
    dilation = 1
    
    # x is 2D instead of 3D.
    x = torch.randn(in_channels, input_length, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels, kernel_size, device="cuda", dtype=torch.float32)
    
    with pytest.raises(RuntimeError):
        cuda_module.forward(x, weight, None, stride, padding, dilation)


# Test 4: Large output tensor may reveal grid-stride inefficiencies (or potential issues)
def test_large_output():
    cuda_module = build_kernel()
    batch_size = 4
    in_channels = 8
    out_channels = 16
    kernel_size = 5
    input_length = 64  # moderate input length
    stride = 4
    padding = 2
    dilation = 2

    x = torch.randn(batch_size, in_channels, input_length, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    # Run the kernel.  This test does not compare against a reference numerical value,
    # but we expect the kernel to complete without errors. In an inefficient implementation,
    # scheduling issues or silent errors might occur.
    output = cuda_module.forward(x, weight, bias, stride, padding, dilation)
    assert output.shape[0] == batch_size
    # Compute expected output length using the same formula as in the kernel.
    expected_output_length = (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    assert output.shape[2] == expected_output_length

