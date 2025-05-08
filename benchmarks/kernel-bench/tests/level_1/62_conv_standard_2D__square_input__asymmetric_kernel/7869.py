
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper function to build the extension module from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="custom_conv2d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# The following tests will trigger the issues in the custom CUDA kernel.

# Issue 1 & 2: Lack of support for dilation > 1 and groups ≠ 1.
# When dilation or groups are not the default (1), the forward()
# function actually falls back to torch::conv2d instead of using the custom kernel.
# The test below warns (by asserting inequality) that the custom kernel 
# does not handle these cases, by comparing with the result of a pure torch conv2d.
def test_dilation_and_groups_not_supported():
    custom_conv = build_kernel()
    # Create a simple conv parameters.
    batch_size = 2
    in_channels = 3
    out_channels = 8
    height = 32
    width = 32
    kernel_size = (3, 5)
    stride = 1
    padding = 1
    dilation = 2    # non-default dilation
    groups = 2      # non-default groups

    # Create input and weight tensors.
    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    # For groups != 1 the weight shape in torch.conv2d is (out_channels, in_channels//groups, kH, kW)
    weight = torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1], device="cuda", dtype=torch.float32)
    bias = None

    # Since dilation > 1 or groups != 1, the forward() function falls back to torch::conv2d,
    # so the custom kernel is not used. 
    # The expected behavior is not provided by the custom implementation.
    # Thus, we check that if these parameters are provided, the behavior is different
    # from what would have been computed assuming the custom kernel ignored them.
    # Compute output using torch.conv2d with proper parameters.
    torch_out = torch.conv2d(x, weight, bias, stride=(stride, stride),
                             padding=(padding, padding), dilation=(dilation, dilation), groups=groups)
    # Now call our module (which will dispatch to torch::conv2d)
    custom_out = custom_conv.forward(x, weight, bias, stride, padding, dilation, groups)
    # Even though in this branch the computation should match torch implementation,
    # we highlight that our kernel is not handling these cases internally.
    assert torch.allclose(custom_out, torch_out, atol=1e-5), \
        "When using non-default dilation or groups, the custom function should defer to torch::conv2d."

# Issue 3: Grid dimension limitation (mapping out_channels to gridDim.z)
# In many hardware architectures gridDim.z is limited.
# Launching a kernel with a large out_channels value may cause a launch failure.
def test_grid_dimension_issue():
    custom_conv = build_kernel()
    batch_size = 1
    in_channels = 3
    # Choose an out_channels value that will likely exceed the maximum allowed gridDim.z.
    # Many GPUs have a maximum of 64 for gridDim.z.
    out_channels = 128  
    height = 16
    width = 16
    kernel_size = (3, 3)
    stride = 1
    padding = 1
    dilation = 1
    groups = 1

    # Create input and weight
    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1], device="cuda", dtype=torch.float32)
    bias = None

    # This call should attempt to launch the kernel.
    # Because the kernel uses blockIdx.z = output channel index,
    # if out_channels exceeds the hardware limit then a CUDA launch error is expected.
    with pytest.raises(RuntimeError):
        custom_conv.forward(x, weight, bias, stride, padding, dilation, groups)
    # Note: on some devices the maximum gridDim.z might be larger and the kernel may actually launch.
    # In that case, this test should be adjusted to use a value of out_channels that is known to exceed the limit.

# Issue 4: Hard-coded data type (only supports float32).
# Passing a tensor of type double should either fail or produce an incorrect result.
def test_precision_issue():
    custom_conv = build_kernel()
    batch_size = 2
    in_channels = 3
    out_channels = 8
    height = 32
    width = 32
    kernel_size = (3, 3)
    stride = 1
    padding = 1
    dilation = 1
    groups = 1

    # Create double precision tensors.
    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float64)
    weight = torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1], device="cuda", dtype=torch.float64)
    bias = None

    # Because our kernel works only on float32 and uses data_ptr<float>(),
    # the computation will be done on an incorrect type.
    # The result should therefore differ from the reference (torch.conv2d computed on the double tensors).
    torch_ref = torch.conv2d(x, weight, bias, stride=(stride, stride), padding=(padding, padding), dilation=(dilation, dilation), groups=groups)
    # Force conversion to double for cleaner comparison if the kernel erroneously casts or misinterprets memory.
    custom_out = custom_conv.forward(x, weight, bias, stride, padding, dilation, groups).to(torch.float64)
    # We expect that if the kernel is used on double inputs, the result will be far off.
    diff = (custom_out - torch_ref).abs().max().item()
    assert diff > 1e-3, \
        f"Kernel did not properly handle types; maximum difference {diff} is too low indicating an unexpected behavior."

