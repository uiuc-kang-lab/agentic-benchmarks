
import torch
import pytest
from torch.nn import ConvTranspose2d
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="custom_conv_transpose2d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1 and 3: Groups parameter is ignored and weight tensor layout for grouped convolution is not handled.
def test_groups_not_supported():
    # Use groups != 1 to trigger the issue.
    batch_size = 4
    in_channels = 4
    out_channels = 8
    kernel_size = 3
    stride = 1
    padding = 1
    output_padding = 0
    groups = 2  # Use groups > 1

    # Create input tensor
    x = torch.randn(batch_size, in_channels, 16, 16, device="cuda", dtype=torch.float32)
    # Create conv_transpose2d with groups
    conv = ConvTranspose2d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding,
        output_padding=output_padding, groups=groups, bias=True
    ).cuda()

    # Get weight and bias from conv model.
    weight = conv.weight.contiguous()
    bias = conv.bias.contiguous()
    # Compute reference output using PyTorch's conv_transpose2d
    torch_ref = conv(x)

    # Use the custom kernel forward.
    # The custom kernel expects weight shape [in_channels, out_channels, kernel, kernel]
    # but for groups != 1, torch conv_transpose2d uses weight shape [in_channels, out_channels//groups, kernel, kernel]
    # So we intentionally "fix" the shape to the one expected by the kernel to trigger the issue.
    # Here we simply pass the same weight, so the extension is unaware of groups.
    cuda_module = build_kernel()
    torch_cuda = cuda_module.forward(x, weight, bias, stride, padding, output_padding, groups)

    # They should NOT match due to groups being ignored.
    assert not torch.allclose(torch_ref, torch_cuda, atol=1e-4), \
        "Test failed to trigger groups issue: outputs unexpectedly match."

# Issue 2: Asymmetric stride and related parameters are not supported.
def test_asymmetric_parameters_not_supported():
    batch_size = 4
    in_channels = 3
    out_channels = 6
    kernel_size = 3
    # Provide asymmetric stride and padding as tuples.
    stride = (1, 2)
    padding = (0, 1)
    output_padding = (1, 0)
    groups = 1

    # Create input tensor with asymmetric spatial dimensions.
    x = torch.randn(batch_size, in_channels, 10, 12, device="cuda", dtype=torch.float32)
    # Create conv_transpose2d with asymmetric arguments.
    conv = ConvTranspose2d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding,
        output_padding=output_padding, groups=groups, bias=False
    ).cuda()

    # Reference output using PyTorch
    torch_ref = conv(x)

    # Use our custom kernel.
    cuda_module = build_kernel()
    torch_cuda = cuda_module.forward(x, conv.weight.contiguous(), None, stride, padding, output_padding, groups)

    # Since the kernel only takes the first element of each tuple parameter,
    # the output shape computed will differ from the correct one.
    expected_shape = torch_ref.shape
    cuda_shape = torch_cuda.shape
    assert cuda_shape != expected_shape, \
        f"Test failed to trigger asymmetric parameters issue: expected shape {expected_shape} but got {cuda_shape}."

if __name__ == "__main__":
    pytest.main([__file__])
