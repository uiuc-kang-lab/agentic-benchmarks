
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="conv_transpose2d_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_output_padding_issue():
    # This test is designed to trigger issue 1: the kernel fails to apply output_padding in computation.
    batch_size = 1
    in_channels = 4
    out_channels = 8
    kernel_size = (3, 3)
    stride = 2
    padding = 1
    output_padding = 1  # nonzero output_padding
    groups = 1

    x = torch.randn(batch_size, in_channels, 10, 10, device='cuda', dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels // groups, kernel_size[0], kernel_size[1],
                         device='cuda', dtype=torch.float32)
    bias = None

    # Use PyTorch’s built-in ConvTranspose2d as reference.
    ref_conv = torch.nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding, output_padding=output_padding,
        groups=groups, bias=False
    ).cuda()
    ref_conv.weight.data = weight.clone()
    ref_out = ref_conv(x)

    mod = build_kernel()
    kernel_out = mod.forward(x, weight, bias, stride, padding, output_padding, groups, 1)

    # Due to the missing handling of output_padding in the CUDA kernel,
    # the kernel output should differ from the reference.
    assert not torch.allclose(kernel_out, ref_out, atol=1e-4), (
        "Kernel output unexpectedly matches reference output. "
        "Output_padding issue may have been fixed (or not triggered)."
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_asymmetric_stride_padding_issue():
    # This test is designed to trigger issue 2: the kernel only accepts scalar stride/padding.
    # Here we compare a built-in ConvTranspose2d (using different stride/padding per dim) with our kernel.
    # Since our kernel only accepts scalar values, a mismatch in behavior is expected.
    batch_size = 1
    in_channels = 3
    out_channels = 6
    kernel_size = (3, 5)  # asymmetric kernel shape
    # For the built-in layer, we use differing strides/paddings per dimension.
    ref_stride = (2, 3)
    ref_padding = (1, 2)
    ref_output_padding = (1, 0)
    groups = 1

    x = torch.randn(batch_size, in_channels, 8, 8, device='cuda', dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels // groups, kernel_size[0], kernel_size[1],
                         device='cuda', dtype=torch.float32)
    bias = None

    ref_conv = torch.nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size,
        stride=ref_stride, padding=ref_padding, output_padding=ref_output_padding,
        groups=groups, bias=False
    ).cuda()
    ref_conv.weight.data = weight.clone()
    ref_out = ref_conv(x)

    mod = build_kernel()
    # Our kernel can only accept scalars. We supply the first element of the tuple parameters.
    kernel_out = mod.forward(x, weight, bias, ref_stride[0], ref_padding[0], ref_output_padding[0], groups, 1)

    # The outputs will differ because our kernel cannot honor asymmetric stride/padding.
    assert not torch.allclose(kernel_out, ref_out, atol=1e-4), (
        "Kernel output unexpectedly matches reference output for asymmetric stride/padding."
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_transpose_indexing_issue():
    # This test is designed to trigger issue 3: the mapping from output indices to input positions is incorrect.
    # Using a dilation > 1 will expose the error in index computations.
    batch_size = 1
    in_channels = 4
    out_channels = 4
    kernel_size = (3, 3)
    stride = 2
    padding = 1
    output_padding = 0
    dilation = 2  # nontrivial dilation exposes indexing errors
    groups = 1

    x = torch.randn(batch_size, in_channels, 12, 12, device='cuda', dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels // groups, kernel_size[0], kernel_size[1],
                         device='cuda', dtype=torch.float32)
    bias = None

    ref_conv = torch.nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding, output_padding=output_padding,
        dilation=dilation, groups=groups, bias=False
    ).cuda()
    ref_conv.weight.data = weight.clone()
    ref_out = ref_conv(x)

    mod = build_kernel()
    kernel_out = mod.forward(x, weight, bias, stride, padding, output_padding, groups, dilation)
    assert not torch.allclose(kernel_out, ref_out, atol=1e-4), (
        "Kernel output unexpectedly matches reference output for dilation test. "
        "Transpose indexing issue may not have been triggered."
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_non_contiguous_tensor_issue():
    # This test is designed to trigger issue 4: non-contiguous inputs are not handled.
    # We compare the result for a non-contiguous tensor with that for a contiguous one.
    batch_size, in_channels, H, W = 1, 3, 15, 15
    x = torch.randn(batch_size, in_channels, H, W, device='cuda', dtype=torch.float32)
    # Create a non-contiguous version of x.
    x_non_contig = x.permute(0, 2, 3, 1)
    # Note: weight is created normally (contiguous).
    weight = torch.randn(in_channels, 4, 3, 3, device='cuda', dtype=torch.float32)
    bias = None

    mod = build_kernel()
    # Run kernel with the non-contiguous input.
    kernel_out_non_contig = mod.forward(x_non_contig, weight, bias, 1, 1, 0, 1, 1)
    # Also run kernel with a contiguous copy.
    kernel_out_contig = mod.forward(x_non_contig.contiguous(), weight, bias, 1, 1, 0, 1, 1)
    # If the kernel assumes contiguous tensors, the outputs may differ.
    assert not torch.allclose(kernel_out_non_contig, kernel_out_contig, atol=1e-4), (
        "Non-contiguous tensor usage did not trigger a difference in output."
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_input_device_error():
    # This test is designed to trigger a runtime error when the input is not a CUDA tensor.
    x_cpu = torch.randn(1, 3, 8, 8, device='cpu', dtype=torch.float32)
    weight_cpu = torch.randn(3, 4, 3, 3, device='cpu', dtype=torch.float32)
    mod = build_kernel()
    with pytest.raises(RuntimeError):
        mod.forward(x_cpu, weight_cpu, None, 1, 0, 0, 1, 1)
