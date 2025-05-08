
import torch
import pytest
from torch.utils.cpp_extension import load

# Function to build and load the CUDA kernel extension.
def build_kernel():
    cuda_module = load(
        name="custom_conv_transpose_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# -----------------------------------------------------------------------------
# Test to trigger Issue 1: output_padding is ignored.
# We create a standard nn.ConvTranspose2d model and compare its output against
# the custom kernel output when output_padding is non-zero.
# The custom kernel will compute a different shape (or values) than expected.
def test_output_padding_issue():
    batch_size = 2
    in_channels = 4
    out_channels = 6
    kernel_size = 3
    stride = 2
    padding = 1
    output_padding = 1  # non-zero output padding
    groups = 1
    bias = False

    # Create input
    x = torch.randn(batch_size, in_channels, 16, 16, device="cuda", dtype=torch.float32)
    
    # Build the standard ConvTranspose2d (which uses correct output_padding)
    standard_convT = torch.nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding, output_padding=output_padding,
        groups=groups, bias=bias
    ).cuda()
    
    # Obtain standard output
    y_std = standard_convT(x)

    # Prepare parameters for custom kernel function.
    # For testing purposes we mimic the weight layout:
    # standard weight shape: [in_channels, out_channels/groups, kernel_size, kernel_size]
    weight = standard_convT.weight.detach()
    bias_tensor = standard_convT.bias.detach() if bias else torch.tensor([], device="cuda")
    
    # Call custom kernel: note that our kernel function does not use output_padding.
    custom_module = build_kernel()
    y_custom = custom_module.forward(
        x,
        weight,
        bias_tensor if bias else torch.tensor([], device="cuda"),
        stride,
        padding,
        output_padding,  # provided but ignored inside the kernel.
        groups
    )
    # Because the kernel ignores output_padding, the output shape will differ.
    assert y_std.shape != y_custom.shape, (
        "Custom kernel unexpectedly produced the same shape as standard ConvTranspose2d. "
        "This indicates that output_padding is not being applied correctly."
    )

# -----------------------------------------------------------------------------
# Test to trigger Issue 2: Non-square kernel not supported.
# We simulate a non-square kernel by manually constructing a weight tensor with different
# dimensions for height and width and check that the custom kernel does not work correctly.
def test_nonsquare_kernel_issue():
    batch_size = 2
    in_channels = 4
    out_channels = 6
    kernel_size_h = 3
    kernel_size_w = 5  # non-square kernel
    stride = 1
    padding = 0
    output_padding = 0
    groups = 1
    bias = False

    x = torch.randn(batch_size, in_channels, 16, 16, device="cuda", dtype=torch.float32)
    
    # Manually create a weight tensor with non-square kernel dimensions.
    # Even though standard PyTorch modules do not allow non-square kernels for ConvTranspose2d,
    # we mimic such a situation to test if the custom kernel (which hard-codes square kernels) fails.
    weight = torch.randn(in_channels, out_channels // groups, kernel_size_h, kernel_size_w, device="cuda", dtype=torch.float32)
    bias_tensor = torch.empty(0, device="cuda")  # no bias

    custom_module = build_kernel()
    with pytest.raises(RuntimeError):
        # The custom kernel assumes weight has shape [in_channels, out_channels/groups, K, K]
        # and uses weight.size(2) as square kernel size. The discrepancy should trigger an error,
        # or at least yield an output that does not match the expected computation.
        _ = custom_module.forward(x, weight, bias_tensor, stride, padding, output_padding, groups)

# -----------------------------------------------------------------------------
# Test to trigger Issue 3: Incorrect CUDA stream usage.
# We create a non-default stream and launch the kernel; if cudaStreamDefault is used inside,
# synchronization issues may arise. In this test we check that the computation does not occur
# on the correct stream.
def test_cuda_stream_issue():
    batch_size = 2
    in_channels = 4
    out_channels = 4
    kernel_size = 3
    stride = 1
    padding = 1
    output_padding = 0
    groups = 1
    bias = False

    x = torch.randn(batch_size, in_channels, 16, 16, device="cuda", dtype=torch.float32)
    
    # Instantiate a ConvTranspose2d module to get weight and bias.
    convT = torch.nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size, stride=stride, padding=padding,
        output_padding=output_padding, groups=groups, bias=bias
    ).cuda()
    weight = convT.weight.detach()
    bias_tensor = convT.bias.detach() if bias else torch.tensor([], device="cuda")

    # Create a non-default stream.
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        custom_module = build_kernel()
        y_custom = custom_module.forward(x, weight, bias_tensor, stride, padding, output_padding, groups)
        # Intentionally not synchronizing with the non-default stream may lead to errors if the default
        # stream is incorrectly used in the kernel launch.
    # Synchronize default stream and check shape.
    torch.cuda.synchronize()
    # We expect the default and non-default streams to not interfere.
    # Here, if the kernel was launched on cudaStreamDefault instead of the current stream, the test may fail
    # in more complex asynchronous scenarios. We only check that the output has the expected shape.
    expected_height = (x.size(2) - 1) * stride - 2 * padding + kernel_size + output_padding
    expected_width = (x.size(3) - 1) * stride - 2 * padding + kernel_size + output_padding
    assert y_custom.shape == (batch_size, out_channels, expected_height, expected_width), (
        "Output shape does not match expected shape, potentially due to incorrect stream usage."
    )

# -----------------------------------------------------------------------------
# Test to trigger Issue 4: Only float32 is supported.
# We create an input tensor of a different data type (float64) and expect the kernel to raise an error.
def test_dtype_issue():
    batch_size = 2
    in_channels = 4
    out_channels = 4
    kernel_size = 3
    stride = 1
    padding = 1
    output_padding = 0
    groups = 1
    bias = False

    # Create input tensor in float64.
    x = torch.randn(batch_size, in_channels, 16, 16, device="cuda", dtype=torch.float64)
    
    convT = torch.nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size, stride=stride, padding=padding,
        output_padding=output_padding, groups=groups, bias=bias
    ).cuda()
    # Convert weight to float64
    weight = convT.weight.detach().to(torch.float64)
    bias_tensor = convT.bias.detach().to(torch.float64) if bias else torch.tensor([], device="cuda", dtype=torch.float64)
    
    custom_module = build_kernel()
    with pytest.raises(RuntimeError):
        _ = custom_module.forward(x, weight, bias_tensor, stride, padding, output_padding, groups)

# -----------------------------------------------------------------------------
# Test to trigger Issue 5: Output alignment assumptions.
# While we cannot force misaligned allocation easily in PyTorch,
# we can check the pointer alignment of the output tensor.
def test_output_alignment_issue():
    batch_size = 2
    in_channels = 4
    out_channels = 4
    kernel_size = 3
    stride = 1
    padding = 1
    output_padding = 0
    groups = 1
    bias = False

    x = torch.randn(batch_size, in_channels, 16, 16, device="cuda", dtype=torch.float32)
    
    convT = torch.nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size, stride=stride, padding=padding,
        output_padding=output_padding, groups=groups, bias=bias
    ).cuda()
    weight = convT.weight.detach()
    bias_tensor = convT.bias.detach() if bias else torch.tensor([], device="cuda")
    
    custom_module = build_kernel()
    y_custom = custom_module.forward(x, weight, bias_tensor, stride, padding, output_padding, groups)
    
    # Check pointer alignment: 128-bit alignment means the pointer should be a multiple of 16 bytes.
    ptr = y_custom.data_ptr()
    if ptr % 16 != 0:
        pytest.fail("Output tensor is not 128-bit aligned as assumed by the kernel.")

