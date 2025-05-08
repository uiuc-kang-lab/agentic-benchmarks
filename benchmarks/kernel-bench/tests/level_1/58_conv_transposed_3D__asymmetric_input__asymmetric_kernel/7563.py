
import torch
import pytest
from torch.utils.cpp_extension import load
import torch.nn as nn

def build_kernel():
    cuda_module = load(
        name="custom_conv_transpose3d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_output_padding_issue():
    # This test uses non-zero output_padding.
    # We compare against a reference nn.ConvTranspose3d implementation.
    N = 2
    in_channels = 3
    out_channels = 4
    kernel_size = (3, 3, 3)
    stride = (2, 2, 2)
    padding = (1, 1, 1)
    output_padding = (1, 0, 1)  # non-zero output_padding to trigger the issue
    groups = 1

    # Create input and a ConvTranspose3d layer
    input_tensor = torch.randn(N, in_channels, 4, 4, 4, device='cuda', dtype=torch.float32)
    conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, output_padding=output_padding, bias=False).cuda()
    # Retrieve the reference output from the PyTorch implementation
    ref_output = conv(input_tensor)

    # Extract the weight from conv and use it for the custom kernel
    weight = conv.weight.detach().clone()

    kernel_module = build_kernel()
    # Pass an empty Tensor for bias if not provided (this mimics bias not used)
    output_kernel = kernel_module.forward(
        input_tensor,
        weight,
        torch.Tensor(),  # no bias provided
        list(stride),
        list(padding),
        list(output_padding),
        groups
    )
    # Since the kernel ignores output_padding in its inner loop,
    # the output from the custom kernel should differ from the reference.
    assert not torch.allclose(ref_output, output_kernel, atol=1e-4), \
        "Kernel appears to handle output_padding correctly, but it should ignore it (issue not triggered)."

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_dtype_issue():
    # This test uses float64 tensors.
    # The kernel is hardcoded for float32; hence, using float64 should raise an error.
    N = 2
    in_channels = 3
    out_channels = 4
    kernel_size = (3, 3, 3)
    stride = (1, 1, 1)
    padding = (0, 0, 0)
    output_padding = (0, 0, 0)
    groups = 1

    input_tensor = torch.randn(N, in_channels, 5, 5, 5, device="cuda", dtype=torch.float64)
    weight = torch.randn(in_channels, out_channels // groups, kernel_size[0], kernel_size[1], kernel_size[2],
                         device="cuda", dtype=torch.float64)
    kernel_module = build_kernel()
    with pytest.raises(RuntimeError):
        # The kernel expects float32 data; using float64 should trigger a runtime error.
        _ = kernel_module.forward(
            input_tensor,
            weight,
            torch.Tensor(),  # no bias provided
            list(stride),
            list(padding),
            list(output_padding),
            groups
        )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_non_contiguous_issue():
    # This test creates a non-contiguous input tensor.
    # The kernel assumes contiguous memory; non-contiguous inputs may produce incorrect results.
    N = 1
    in_channels = 4
    out_channels = 4
    kernel_size = (3, 3, 3)
    stride = (1, 1, 1)
    padding = (1, 1, 1)
    output_padding = (0, 0, 0)
    groups = 1

    input_tensor = torch.randn(N, in_channels, 6, 6, 6, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels // groups, kernel_size[0], kernel_size[1], kernel_size[2],
                         device="cuda", dtype=torch.float32)
    # Make input non-contiguous (e.g., by transposing two dimensions)
    input_noncontiguous = input_tensor.transpose(1, 2)
    kernel_module = build_kernel()
    out_contiguous = kernel_module.forward(
        input_tensor,
        weight,
        torch.Tensor(),  # no bias provided
        list(stride),
        list(padding),
        list(output_padding),
        groups
    )
    out_noncontiguous = kernel_module.forward(
        input_noncontiguous,
        weight,
        torch.Tensor(),  # no bias provided
        list(stride),
        list(padding),
        list(output_padding),
        groups
    )
    # If the kernel does not account for non-contiguous memory, the results will differ.
    assert not torch.allclose(out_contiguous, out_noncontiguous, atol=1e-4), \
        "Kernel incorrectly handles non-contiguous inputs; outputs match when they should differ."
