
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="custom_conv_transpose",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

def test_groups_issue():
    # Test using groups != 1.
    # For groups=2, nn.ConvTranspose2d expects weight of shape [in_channels, out_channels//2, k, k],
    # but our kernel always assumes weight shape [in_channels, out_channels, k, k].
    device = "cuda"
    batch_size = 2
    in_channels = 4
    out_channels = 4  # With groups=2, proper weight shape would be (4,2,k,k)
    kernel_size = 3

    x = torch.randn(batch_size, in_channels, 8, 8, device=device)
    # Create a weight tensor with the proper shape for reference: [in_channels, out_channels//groups, k, k]
    ref_weight = torch.randn(in_channels, out_channels // 2, kernel_size, kernel_size, device=device)
    bias = torch.randn(out_channels, device=device)

    # Construct the reference conv_transpose2d with groups=2.
    ref_conv = torch.nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size,
        stride=1, padding=0, output_padding=0, groups=2, bias=True
    ).to(device)
    with torch.no_grad():
        ref_conv.weight.copy_(ref_weight)
        ref_conv.bias.copy_(bias)
    ref_output = ref_conv(x)

    # Prepare a weight tensor for our custom kernel.
    # Our kernel expects weight shape [in_channels, out_channels, k, k].
    # Simulate a workaround by embedding the group weights in a larger tensor.
    weight_custom = torch.zeros(in_channels, out_channels, kernel_size, kernel_size, device=device)
    for g in range(2):
        start_in = g * (in_channels // 2)
        end_in = (g + 1) * (in_channels // 2)
        start_out = g * (out_channels // 2)
        end_out = (g + 1) * (out_channels // 2)
        weight_custom[start_in:end_in, start_out:end_out] = ref_weight[start_in:end_in]
    
    custom_mod = build_kernel()
    custom_output = custom_mod.forward(
        x, weight_custom, bias, stride=1, padding=0, output_padding=0, groups=2
    )
    
    # Due to ignoring the groups parameter, the custom output will be different from the reference.
    assert not torch.allclose(custom_output, ref_output, atol=1e-4), \
        "Groups issue not detected: custom kernel matches reference unexpectedly."

def test_tuple_stride_padding_issue():
    # Test using tuple values for stride, padding, and output_padding.
    # The custom kernel only uses the first element of each tuple.
    device = "cuda"
    batch_size = 2
    in_channels = 3
    out_channels = 3
    kernel_size = 3

    x = torch.randn(batch_size, in_channels, 10, 10, device=device)
    weight = torch.randn(in_channels, out_channels, kernel_size, kernel_size, device=device)
    bias = None

    # Build reference with asymmetric (tuple) parameters.
    ref_conv = torch.nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size,
        stride=(2, 3), padding=(1, 2), output_padding=(0, 1), bias=False
    ).to(device)
    with torch.no_grad():
        ref_conv.weight.copy_(weight)
    ref_output = ref_conv(x)

    custom_mod = build_kernel()
    # Custom kernel will use only the first element of each tuple.
    custom_output = custom_mod.forward(
        x, weight, bias, stride=(2, 3), padding=(1, 2), output_padding=(0, 1), groups=1
    )

    # The output shape should differ because the custom kernel interprets parameters as:
    # stride=2, padding=1, output_padding=0.
    assert custom_output.shape != ref_output.shape, \
        "Tuple stride/padding issue not detected: output shapes match unexpectedly."

def test_dtype_issue():
    # Test behavior with non-float32 (e.g., float64) tensors.
    device = "cuda"
    batch_size = 2
    in_channels = 3
    out_channels = 3
    kernel_size = 3

    x = torch.randn(batch_size, in_channels, 8, 8, device=device, dtype=torch.float64)
    weight = torch.randn(in_channels, out_channels, kernel_size, kernel_size, device=device, dtype=torch.float64)
    bias = None

    custom_mod = build_kernel()
    # We expect the kernel to only support float32, so either an error is raised
    # or the output dtype inadvertently becomes float32.
    try:
        custom_output = custom_mod.forward(x, weight, bias, stride=1, padding=0, output_padding=0, groups=1)
    except Exception as e:
        assert "float" in str(e).lower(), f"Expected a dtype error, but got: {e}"
    else:
        assert custom_output.dtype == torch.float32, \
            "Kernel did not convert non-float32 input to float32 as expected."

def test_weight_shape_issue():
    # Test behavior when the weight tensor is non-square.
    # The kernel expects a square kernel, i.e. shape [in_channels, out_channels, k, k],
    # so providing a weight with shape [in_channels, out_channels, 3, 4] should trigger an error.
    device = "cuda"
    batch_size = 2
    in_channels = 3
    out_channels = 3

    # Non-square kernel: 3 x 4 instead of 3 x 3.
    weight = torch.randn(in_channels, out_channels, 3, 4, device=device)
    x = torch.randn(batch_size, in_channels, 8, 8, device=device)
    bias = None

    custom_mod = build_kernel()
    with pytest.raises(RuntimeError):
        _ = custom_mod.forward(x, weight, bias, stride=1, padding=0, output_padding=0, groups=1)
