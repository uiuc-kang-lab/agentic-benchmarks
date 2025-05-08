
import torch
import pytest
from torch.utils.cpp_extension import load
import torch.nn.functional as F

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1: Weight tensor dimension mismatch.
# We provide a weight tensor with 4 dimensions (the correct PyTorch conv2d weight shape) and compare
# the output from the custom kernel with PyTorch’s nn.functional.conv2d result.
def test_weight_dim_mismatch():
    cuda_module = build_kernel()
    # Use a small input for which the weight misinterpretation is evident.
    batch_size = 2
    in_channels = 3
    height, width = 8, 8
    kernel_size = 3
    stride = 1
    padding = 1
    # Depthwise conv weight for PyTorch: shape (in_channels, 1, kernel_size, kernel_size)
    weight = torch.randn(in_channels, 1, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(in_channels, device="cuda", dtype=torch.float32)
    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)

    # Compute reference output using PyTorch's conv2d with groups=in_channels.
    ref_out = F.conv2d(x, weight, bias, stride=stride, padding=padding, groups=in_channels)

    # Using the custom kernel:
    # Note: our kernel expects weight as a 3D tensor of shape (in_channels, kernel_size, kernel_size).
    # Therefore, simulate an unintended conversion (squeeze the second dimension).
    wrong_weight = weight.squeeze(1)
    custom_out = cuda_module.forward(x, wrong_weight, bias, stride, padding, groups=in_channels)
    
    # The outputs should be different because the kernel misinterprets the weight layout.
    assert not torch.allclose(custom_out, ref_out, atol=1e-5), "Test failed: Kernel output matches reference despite weight dimension mismatch."

# Issue 2: Grid dimension (blockIdx.z) exceeding hardware limit.
# We create an input whose product of batch_size * in_channels exceeds the typical maximum grid size along z.
def test_grid_dimension_limit():
    cuda_module = build_kernel()
    # Choose values such that batch_size * in_channels > 65535.
    # For example, using batch_size = 1024 and in_channels = 100 yields 102400 > 65535,
    # but we keep the spatial dimensions very small to avoid huge memory usage.
    batch_size = 1024
    in_channels = 100
    height, width = 5, 5
    kernel_size = 3
    stride = 1
    padding = 1

    # Create weight tensor of shape (in_channels, 1, kernel_size, kernel_size) then squeeze
    weight = torch.randn(in_channels, 1, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(in_channels, device="cuda", dtype=torch.float32)
    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)

    # The custom kernel expects weight as 3D, so we squeeze the extra dimension.
    wrong_weight = weight.squeeze(1)
    
    with pytest.raises(RuntimeError):
        # This kernel launch should fail because the grid z-dimension would exceed the limit.
        _ = cuda_module.forward(x, wrong_weight, bias, stride, padding, groups=in_channels)
        torch.cuda.synchronize()

# Issue 3: The kernel ignores the groups parameter.
# We provide an input with groups different from in_channels (non-depthwise) and compare to PyTorch's conv2d.
def test_groups_parameter_ignored():
    cuda_module = build_kernel()
    # Use parameters for a non-depthwise convolution.
    # For instance, let in_channels = 4 and groups = 2. In a standard conv2d, this is allowed,
    # but our kernel always treats the convolution as depthwise (groups == in_channels).
    batch_size = 2
    in_channels = 4
    height, width = 8, 8
    kernel_size = 3
    stride = 1
    padding = 1
    groups = 2

    # PyTorch conv2d weight shape is (in_channels, in_channels/groups, kernel_size, kernel_size)
    weight = torch.randn(in_channels, in_channels // groups, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(in_channels, device="cuda", dtype=torch.float32)
    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)

    # Reference output using PyTorch conv2d:
    ref_out = F.conv2d(x, weight, bias, stride=stride, padding=padding, groups=groups)

    # The custom kernel does not use the groups parameter; it assumes groups == in_channels.
    # To force the kernel to run, we simulate a call by reshaping weight to 3D as if it were for depthwise conv.
    # This mismapping will give the wrong result compared to the ref_out.
    # Create a "depthwise-style" weight from the provided weight by summing or selecting along the channel dimension.
    # Here, we simply take the first kernel_weight from each expected channel.
    # (Note: this manipulation is solely to invoke the custom kernel with a 3D weight.)
    depthwise_weight = weight[:, 0, :, :]  # shape: (in_channels, kernel_size, kernel_size)

    custom_out = cuda_module.forward(x, depthwise_weight, bias, stride, padding, groups=groups)
    
    # Because groups is ignored in the kernel, the output should differ from the correct convolution.
    assert not torch.allclose(custom_out, ref_out, atol=1e-5), "Test failed: Kernel output unexpectedly matches reference when groups parameter is not used."
    
if __name__ == "__main__":
    pytest.main([__file__])
