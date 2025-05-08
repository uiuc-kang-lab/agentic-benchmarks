
import pytest
import torch
from torch.utils.cpp_extension import load

# Utility function to build the extension
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1: Missing weight reversal.
# In a correct transposed convolution, the kernel (weight) should be flipped.
# Here, we set a fixed weight and bias to compare the custom kernel versus PyTorch's ConvTranspose1d.
def test_weight_flip():
    torch.manual_seed(0)
    batch_size = 2
    in_channels = 3
    out_channels = 4
    kernel_size = 3
    length = 10
    stride = 2
    padding = 1
    dilation = 1
    bias = False

    # Create input and fixed weights.
    x = torch.randn(batch_size, in_channels, length, device='cuda', dtype=torch.float32)
    # Create weight of shape (in_channels, out_channels, kernel_size)
    weight = torch.randn(in_channels, out_channels, kernel_size, device='cuda', dtype=torch.float32)
    # For simplicity use no bias.
    bias_tensor = None

    # Build module from extension.
    mod = build_kernel()
    out_cuda = mod.forward(x, weight, bias_tensor, stride, padding, dilation)

    # Create a PyTorch reference conv_transpose1d model. Important:
    # PyTorch's ConvTranspose1d uses the weight in the reversed order compared to a direct conv.
    ref_conv = torch.nn.ConvTranspose1d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding, dilation=dilation, bias=bias
    ).cuda()
    # Set the conv layer's weight equal to the test weight.
    # Note: In PyTorch, the weight for ConvTranspose1d is not flipped during assignment.
    with torch.no_grad():
        ref_conv.weight.copy_(weight)
    out_ref = ref_conv(x)

    # Because the custom kernel does not flip the weights, the output will not match.
    assert not torch.allclose(out_cuda, out_ref, atol=1e-4), (
        "Custom kernel output unexpectedly matches PyTorch's conv_transpose1d output. "
        "This indicates that the weight reversal (flip) is performed when it should not be."
    )

# Issue 2: Grouped convolution is not supported.
# We simulate a grouped scenario by manually splitting channels.
def test_groups_not_supported():
    torch.manual_seed(0)
    batch_size = 2
    groups = 2
    in_channels = 4  # Must be divisible by groups.
    out_channels = 4  # For simplicity, set out_channels equal to in_channels.
    kernel_size = 3
    length = 10
    stride = 1
    padding = 1
    dilation = 1
    bias = False

    # Create input and weight for grouped transposed convolution.
    x = torch.randn(batch_size, in_channels, length, device='cuda', dtype=torch.float32)
    # For grouped conv, PyTorch expects the weight shape to be (in_channels, out_channels/groups, kernel_size)
    # but our custom kernel expects (in_channels, out_channels, kernel_size).
    weight = torch.randn(in_channels, out_channels, kernel_size, device='cuda', dtype=torch.float32)

    mod = build_kernel()
    # There is no "groups" parameter in our custom kernel.
    # Therefore, we expect the result to be computed as if groups == 1.
    out_cuda = mod.forward(x, weight, None, stride, padding, dilation)

    # Now, construct a standard grouped ConvTranspose1d and compare.
    ref_conv = torch.nn.ConvTranspose1d(
        in_channels, out_channels, kernel_size, stride=stride,
        padding=padding, dilation=dilation, groups=groups, bias=False
    ).cuda()
    # For reference, split the weight appropriately.
    # This is not a simple copy since our kernel does not expect groups.
    with torch.no_grad():
        # Fill the reference conv weight with a grouped arrangement.
        # For group convolution in ConvTranspose1d, the weight shape is (in_channels, out_channels//groups, kernel_size).
        # We split our weight in half along the out_channels dimension and assign.
        ref_weight = weight.view(groups, in_channels // groups, out_channels, kernel_size)
        # Note: This assignment is only for testing; the results will differ
        # because our extension ignores the groups parameter.
        # Here we simply set the first group weights to be the same.
        ref_conv.weight.copy_(ref_weight.view(in_channels, out_channels // groups, kernel_size))
    out_ref = ref_conv(x)

    # The outputs will differ because our custom kernel treats input as group=1.
    assert not torch.allclose(out_cuda, out_ref, atol=1e-4), (
        "Custom kernel output unexpectedly matches the grouped convolution reference. "
        "This indicates that the kernel does not correctly handle groups."
    )

# Issue 3: The kernel assumes float32 and does not support other data types.
def test_dtype_check():
    torch.manual_seed(0)
    batch_size = 2
    in_channels = 3
    out_channels = 4
    kernel_size = 3
    length = 10
    stride = 1
    padding = 1
    dilation = 1

    # Create input using double precision.
    x = torch.randn(batch_size, in_channels, length, device='cuda', dtype=torch.float64)
    weight = torch.randn(in_channels, out_channels, kernel_size, device='cuda', dtype=torch.float64)

    mod = build_kernel()

    with pytest.raises(RuntimeError):
        # This should raise an error because our kernel expects float32.
        _ = mod.forward(x, weight, None, stride, padding, dilation)

# Issue 4: Lack of CUDA kernel launch error checking.
# We simulate a condition that leads to an invalid kernel launch.
def test_kernel_launch_error():
    torch.manual_seed(0)
    # Here we pass an invalid (excessively large) input size to simulate a kernel launch error.
    batch_size = 1
    in_channels = 3
    out_channels = 4
    kernel_size = 3
    # Use a huge input length that when computing output_length will be enormous.
    length = 1 << 24  
    stride = 1
    padding = 0
    dilation = 1

    x = torch.randn(batch_size, in_channels, length, device='cuda', dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels, kernel_size, device='cuda', dtype=torch.float32)

    mod = build_kernel()
    # We expect that the kernel launch might fail or produce an invalid output.
    # Since the current code does not check for kernel launch errors,
    # we verify that the output does not meet expected properties.
    try:
        out_cuda = mod.forward(x, weight, None, stride, padding, dilation)
        # Force synchronization to trigger potential errors.
        torch.cuda.synchronize()
    except RuntimeError:
        # If a runtime error is thrown, it indicates a kernel launch failure,
        # which is one of the issues we want to detect.
        pytest.skip("Kernel launch error detected as expected; lack of error-checking is an issue.")
    else:
        # If no error is thrown, check that the output size is as expected.
        expected_output_length = (length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
        if out_cuda.numel() != batch_size * out_channels * expected_output_length:
            pytest.fail("Kernel launch error: the output tensor size is incorrect, "
                        "but no error was caught.")

