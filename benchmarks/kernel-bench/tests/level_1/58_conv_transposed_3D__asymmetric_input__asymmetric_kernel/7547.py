
import torch
import pytest
from torch.utils.cpp_extension import load

# Helper function to build the CUDA kernel extension.
def build_kernel():
    cuda_module = load(
        name="custom_transposed_conv3d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1 test: output_padding is ignored.
# We create a small transposed convolution example where output_padding is non-zero.
# We compare the output of PyTorch's built-in nn.ConvTranspose3d with the custom kernel.
def test_output_padding_mismatch():
    # Parameters that force a nonzero output_padding effect.
    batch_size = 2
    in_channels = 4
    out_channels = 6
    kernel_size = (3, 3, 3)
    stride = (2, 2, 2)
    padding = (1, 1, 1)
    output_padding = (1, 0, 0)  # nonzero output_padding on the depth dimension.
    groups = 1
    bias = True

    # Create the PyTorch built-in module.
    conv_transpose = torch.nn.ConvTranspose3d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        bias=bias,
    ).cuda()
    conv_transpose.eval()

    # Prepare input.
    depth_in, height_in, width_in = 8, 8, 8
    x = torch.randn(batch_size, in_channels, depth_in, height_in, width_in, device="cuda")

    # Get PyTorch reference output.
    with torch.no_grad():
        ref_output = conv_transpose(x)

    # Get parameters for the custom kernel.
    # The custom kernel expects weight of shape [in_channels, out_channels/groups, kT, kH, kW].
    weight = conv_transpose.weight.detach()  # shape: [in_channels, out_channels/groups, kT, kH, kW]
    bias_tensor = conv_transpose.bias.detach() if bias else None

    # Build output dimensions following custom kernel host code
    N = x.size(0)
    in_depth = x.size(2)
    in_height = x.size(3)
    in_width = x.size(4)
    kT, kH, kW = weight.size(2), weight.size(3), weight.size(4)
    # Derive out_channels as in custom kernel code.
    out_channels_calc = weight.size(1) * groups
    out_depth = (in_depth - 1) * stride[0] - 2 * padding[0] + kT + output_padding[0]
    out_height = (in_height - 1) * stride[1] - 2 * padding[1] + kH + output_padding[1]
    out_width = (in_width - 1) * stride[2] - 2 * padding[2] + kW + output_padding[2]

    # Build a temporary output tensor (the custom kernel will fill it).
    # The custom kernel expects contiguous tensor of shape (N, out_channels_calc, out_depth, out_height, out_width)
    custom_output = torch.zeros((N, out_channels_calc, out_depth, out_height, out_width), device="cuda", dtype=x.dtype)

    # Build grid parameters for kernel launch (simulate them similar to host code).
    total_output = N * out_channels_calc * out_depth * out_height * out_width
    warpSize = 32
    total_threads = total_output * warpSize
    threads = 256
    blocks = (total_threads + threads - 1) // threads

    # Build the kernel module.
    mod = build_kernel()

    # Dispatch the kernel using AT_DISPATCH_FLOATING_TYPES mechanism from the C++ code.
    # We call the custom "forward" function from our module.
    custom_kernel_output = mod.forward(
        x, weight, bias_tensor,
        list(stride), list(padding), list(output_padding),
        groups
    )

    # The custom kernel ignores output_padding in its index computations.
    # Thus, its results are expected to differ from the PyTorch reference.
    assert not torch.allclose(custom_kernel_output, ref_output, atol=1e-5), \
        "Custom kernel output unexpectedly matches reference output; output_padding seems to be correctly handled!"

# Issue 2 test: integer overflow in index computations.
# We artificially create tensor shapes such that the flattened index multiplication comes close to exceeding the int range.
# (While it is difficult to truly exceed the 32-bit limit on available GPUs, we simulate a scenario where the computed
# indices from the host side become huge.)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_index_overflow():
    # We choose dimensions that are large enough such that the flattened index computation might overflow int.
    # Note: In practice, allocating such huge tensors may be infeasible, so we simulate the scenario with moderately large dims.
    batch_size = 1
    in_channels = 8
    out_channels = 8
    # Use moderately large "depth" dimensions; these numbers are chosen so that their product is large.
    depth_in = 500
    height_in = 500
    width_in = 10  # product = 500*500*10 = 2.5e6 (not overflowing by itself),
                   # but output dimensions are computed by multiplications with stride multiples.
    kernel_size = (3, 3, 3)
    stride = (3, 3, 3)
    padding = (1, 1, 1)
    output_padding = (0, 0, 0)
    groups = 1
    bias = False

    # Setup a built-in conv_transpose3d with these parameters.
    conv_transpose = torch.nn.ConvTranspose3d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        bias=bias,
    ).cuda()
    conv_transpose.eval()

    # Prepare input.
    x = torch.randn(batch_size, in_channels, depth_in, height_in, width_in, device="cuda", dtype=torch.float32)

    with torch.no_grad():
        ref_output = conv_transpose(x)

    weight = conv_transpose.weight.detach()
    bias_tensor = None

    # Compute output dimensions using the host logic from the custom kernel.
    N = x.size(0)
    in_depth = x.size(2)
    in_height = x.size(3)
    in_width = x.size(4)
    kT, kH, kW = weight.size(2), weight.size(3), weight.size(4)
    out_channels_calc = weight.size(1) * groups
    out_depth = (in_depth - 1) * stride[0] - 2 * padding[0] + kT + output_padding[0]
    out_height = (in_height - 1) * stride[1] - 2 * padding[1] + kH + output_padding[1]
    out_width = (in_width - 1) * stride[2] - 2 * padding[2] + kW + output_padding[2]

    mod = build_kernel()
    custom_kernel_output = mod.forward(
        x, weight, bias_tensor,
        list(stride), list(padding), list(output_padding),
        groups
    )

    # Because the kernel uses int for index calculations, the very large computed flat indices might wrap around,
    # producing significantly different output values from the reference.
    # We expect that the outputs will NOT be allclose.
    assert not torch.allclose(custom_kernel_output, ref_output, atol=1e-4), \
        "Custom kernel output unexpectedly matches reference output; potential index overflow issue might not be triggered!"

if __name__ == '__main__':
    pytest.main([__file__])
