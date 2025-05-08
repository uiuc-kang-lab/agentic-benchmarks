
import torch
import torch.nn.functional as F
import pytest
from torch.utils.cpp_extension import load

# Helper to build the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="custom_conv_transpose2d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

##########################
# Issue 1: No valid offset due to unfavorable stride/dilation interaction.
# Explanation: With stride=2 and dilation=4, notice that for any k in [0, stride),
# (k * dilation) % stride is always 0. Hence, if candidate mod stride yields 1 then
# no valid offset is found and the inner accumulation is skipped.
##########################
def test_invalid_offset():
    # Use parameters that force no valid offset for height dimension: 
    # Let stride_h = 2, dilation_h = 4. Then for candidate_h mod 2 == 1 no k satisfies
    # (k * 4) % 2 == 1 because for any k, (k * 4) % 2 == 0.
    batch_size = 1
    in_channels = 2
    out_channels = 2
    kernel_size = (3, 3)
    stride = (2, 2)
    padding = (0, 0)
    dilation = (4, 1)  # Only the height dimension has the problematic combination.
    groups = 1
    bias_flag = False

    # Create a simple input and weight filled with ones to simplify reference computation.
    # Use a fixed seed.
    torch.manual_seed(42)
    x = torch.ones(batch_size, in_channels, 8, 8, device="cuda", dtype=torch.float32)
    weight = torch.ones(in_channels, out_channels // groups, kernel_size[0], kernel_size[1], device="cuda", dtype=torch.float32)
    bias = None  # Let the kernel create a zero bias

    # Build the CUDA kernel extension.
    cuda_ext = build_kernel()

    # Run the custom CUDA kernel.
    output_custom = cuda_ext.forward(x, weight, bias, list(stride), list(padding), list(dilation), groups)

    # Get the reference output using PyTorch's built-in conv_transpose2d.
    # (PyTorch's implementation correctly finds the contributions.)
    output_ref = F.conv_transpose2d(x, weight, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups)

    # Expect a difference due to the loss of accumulation when no valid offset is found.
    # (This test should fail because the custom kernel will produce different output.)
    with pytest.raises(AssertionError):
        torch.testing.assert_allclose(output_custom, output_ref, atol=1e-4)


##########################
# Issue 2: Index overflow for very large tensor shapes.
# Explanation: The kernel uses int for index arithmetic. If the product of dimensions (e.g.
# batch*out_channels*out_h*out_w) exceeds 2**31-1 then integer overflow may occur.
#
# NOTE: It is not practical to allocate an actual tensor of such enormous size, but we simulate
# the situation by calling the kernel with dimensions close to the overflow limit.
##########################
def test_index_overflow(monkeypatch):
    # We simulate very large dimensions by monkeypatching the dimensions passed to the kernel.
    # We keep the actual tensors small so that we do not run out of memory.
    
    batch_size = 1
    in_channels = 1
    out_channels = 1
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = (0, 0)
    dilation = (1, 1)
    groups = 1
    bias_flag = False

    # Create a small input tensor.
    x = torch.ones(batch_size, in_channels, 4, 4, device="cuda", dtype=torch.float32)
    weight = torch.ones(in_channels, out_channels // groups, kernel_size[0], kernel_size[1], device="cuda", dtype=torch.float32)
    bias = None

    # Build the CUDA kernel extension.
    cuda_ext = build_kernel()

    # Monkeypatch the forward function to use huge output dimensions.
    # Here we simulate an overflow by replacing out_h and out_w with large numbers.
    original_forward = cuda_ext.forward

    def forward_overflow(*args, **kwargs):
        # Call original forward to get the small input parameters.
        x_arg, weight_arg, bias_arg, stride_arg, padding_arg, dilation_arg, groups_arg = args
        # Override dimensions to simulate overflow.
        # Set out_h and out_w so that total_elements = batch * out_channels * out_h * out_w exceeds INT_MAX (~2e9)
        large_dim = (2**16)  # 65536; 1 * 1 * 65536 * 65536 = 4294967296 > 2^31-1
        # The forward function in the extension computes out_h based on in_h etc.
        # We temporarily monkeypatch the size() method of x so that x.size(2) and x.size(3) return large dims.
        class FakeSize(torch.Size):
            def __getitem__(self, idx):
                if idx == 2 or idx == 3:
                    return large_dim
                return super().__getitem__(idx)
        fake_size = FakeSize(x_arg.size())
        monkeypatch.setattr(x_arg, "size", lambda idx=None: fake_size if idx is None else fake_size[idx])
        return original_forward(x_arg, weight_arg, bias_arg, stride_arg, padding_arg, dilation_arg, groups_arg)

    monkeypatch.setattr(cuda_ext, "forward", forward_overflow)

    # When running the kernel with these simulated dimensions, the index arithmetic in the kernel
    # might wrap-around. Although we cannot compute a reference result here, we expect strong deviation
    # from the behavior of a correct computation. Thus, the test should catch that the output is wrong.
    output_custom = cuda_ext.forward(x, weight, bias, list(stride), list(padding), list(dilation), groups)
    
    # We use the small tensor computation as a “safe” reference,
    # and check that the simulated overflow output deviates from the safe computation.
    output_safe = F.conv_transpose2d(torch.ones(batch_size, in_channels, 4, 4, device="cuda", dtype=torch.float32),
                                     weight, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups)
    
    # Because of overflow the output should be very different.
    with pytest.raises(AssertionError):
        torch.testing.assert_allclose(output_custom, output_safe, atol=1e-4)


##########################
# Issue 3: Lack of support for true asymmetric padding/output_padding.
# Explanation: The kernel’s documentation touts support for “asymmetric … padded” inputs,
# yet it only accepts a single padding value per spatial dimension and has no parameter for output_padding.
# When these more general cases are required the output will differ from PyTorch’s correct computation.
##########################
def test_asymmetric_padding():
    # Here we simulate the need for output_padding by comparing with the PyTorch implementation.
    batch_size = 1
    in_channels = 2
    out_channels = 2
    kernel_size = (3, 5)
    stride = (2, 3)
    padding = (1, 2)  # These are the only padding values allowed by our kernel.
    dilation = (1, 1)
    groups = 1
    bias_flag = False

    # For nn.ConvTranspose2d, one can pass an output_padding. Our custom kernel does not support it.
    output_padding = (1, 0)  # make padding asymmetric at the output

    torch.manual_seed(123)
    x = torch.randn(batch_size, in_channels, 8, 8, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels // groups, kernel_size[0], kernel_size[1], device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    cuda_ext = build_kernel()
    # Call our custom CUDA kernel (note: it does not accept output_padding)
    output_custom = cuda_ext.forward(x, weight, bias, list(stride), list(padding), list(dilation), groups)

    # Compute reference output with asymmetric output_padding using PyTorch's ConvTranspose2d.
    conv_transpose = torch.nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
        bias=True
    ).to("cuda")
    # Manually override the weight and bias so they match our input.
    conv_transpose.weight.data.copy_(weight)
    conv_transpose.bias.data.copy_(bias)
    output_ref = conv_transpose(x)

    # Since our kernel does not support output_padding (nor asymmetric padding),
    # there should be a discrepancy with the reference.
    with pytest.raises(AssertionError):
        torch.testing.assert_allclose(output_custom, output_ref, atol=1e-4)
