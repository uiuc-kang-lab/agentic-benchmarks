
import torch
import pytest
from torch.utils.cpp_extension import load

# Helper function to compile and load the CUDA extension.
def build_kernel():
    return load(
        name="custom_transposed_conv3d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )

# A helper reference implementation using PyTorch's built-in ConvTranspose3d.
class RefConvTranspose3d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, output_padding, groups, bias):
        super().__init__()
        self.conv_transpose3d = torch.nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
        )
    def forward(self, x):
        return self.conv_transpose3d(x)

# Issue 1: Race condition between convolution and bias addition kernels.
# This test sets up a scenario with non-zero bias and non-trivial inputs.
# Because of the race condition, the results from the CUDA kernel are expected to deviate
# from the reference implementation.
def test_race_condition():
    # Parameters
    batch_size = 2
    in_channels = 4
    out_channels = 8
    kernel_size = (3, 3, 3)
    stride = (2, 2, 2)
    padding = (1, 1, 1)
    output_padding = (1, 1, 1)
    groups = 2
    bias_flag = True

    device = torch.device("cuda")
    torch.manual_seed(42)
    # Use a fixed input and weights so that the expected output is deterministic.
    x = torch.randn(batch_size, in_channels, 8, 8, 8, device=device, dtype=torch.float32)

    # Create a reference model and set identical parameters.
    ref_model = RefConvTranspose3d(in_channels, out_channels, kernel_size,
                                   stride, padding, output_padding, groups, bias_flag).to(device)
    # Synchronize initialization between both implementations by copying parameters.
    weight = ref_model.conv_transpose3d.weight.detach().clone()
    bias_tensor = ref_model.conv_transpose3d.bias.detach().clone() if bias_flag else None

    # Load custom kernel extension.
    custom_kernel = build_kernel()

    # Call our CUDA extension kernel function.
    # Note: The custom kernel is expected to be called as follows:
    #   custom_kernel.forward(input, weight, bias, stride, padding, output_padding, groups)
    # For testing, we assume that the extension's function interface matches the one in the provided code.
    out_custom = custom_kernel.forward(x, weight, bias_tensor, 
                                       [stride[0], stride[1], stride[2]],
                                       [padding[0], padding[1], padding[2]],
                                       [output_padding[0], output_padding[1], output_padding[2]],
                                       groups)

    # Compute the expected output using PyTorch's implementation.
    out_ref = ref_model(x)

    # Race conditions may result in output differences; we expect the maximum absolute difference to be nonzero.
    diff = (out_custom - out_ref).abs().max().item()
    # We set a threshold below which the kernels would be considered in sync. Since we expect a race,
    # the difference should be significant.
    assert diff > 1e-3, f"Expected a significant difference due to race conditions, but diff={diff}"

# Issue 2: Lack of type checking for input tensors.
# If a non-float32 tensor is passed, the kernel will misinterpret the data.
def test_dtype_mismatch():
    # Parameters (similar to the first test, but using double)
    batch_size = 2
    in_channels = 4
    out_channels = 8
    kernel_size = (3, 3, 3)
    stride = (2, 2, 2)
    padding = (1, 1, 1)
    output_padding = (1, 1, 1)
    groups = 2
    bias_flag = True

    device = torch.device("cuda")
    torch.manual_seed(123)
    # Create input of type double instead of float32.
    x = torch.randn(batch_size, in_channels, 8, 8, 8, device=device, dtype=torch.float64)

    # Create a reference model and copy parameters.
    ref_model = RefConvTranspose3d(in_channels, out_channels, kernel_size,
                                   stride, padding, output_padding, groups, bias_flag).to(device)
    # Convert parameters to double.
    weight = ref_model.conv_transpose3d.weight.detach().clone().to(torch.float64)
    bias_tensor = (ref_model.conv_transpose3d.bias.detach().clone().to(torch.float64)
                   if bias_flag else None)

    # Load custom kernel extension.
    custom_kernel = build_kernel()

    # Run the custom kernel (which assumes float32 input internally).
    out_custom = custom_kernel.forward(
        x, weight, bias_tensor,
        [stride[0], stride[1], stride[2]],
        [padding[0], padding[1], padding[2]],
        [output_padding[0], output_padding[1], output_padding[2]],
        groups
    )
    # For a fair comparison, compute the reference output in double precision.
    out_ref = ref_model(x)

    # Since the kernel misinterprets the data (double data passed as if it were float),
    # the output is expected to be very different from the reference.
    diff = (out_custom.to(torch.float64) - out_ref).abs().max().item()
    assert diff > 1e-2, f"Expected significant difference due to dtype mismatch, but diff={diff}"

# Issue 3: Fixed block size may not suit all cases.
# This test attempts a case where the number of input elements is not a multiple of 256
# to check whether any out-of-bound or performance issues occur.
def test_non_multiple_of_block_size():
    # Parameters where input element count is not a multiple of 256.
    batch_size = 1
    in_channels = 3
    out_channels = 6
    kernel_size = (3, 3, 3)
    stride = (1, 1, 1)
    padding = (1, 1, 1)
    output_padding = (0, 0, 0)
    groups = 1
    bias_flag = False

    device = torch.device("cuda")
    torch.manual_seed(999)
    # Choose spatial dimensions such that total input elements (1*3*5*7*9) is not a multiple of 256.
    x = torch.randn(batch_size, in_channels, 5, 7, 9, device=device, dtype=torch.float32)

    ref_model = RefConvTranspose3d(in_channels, out_channels, kernel_size,
                                   stride, padding, output_padding, groups, bias_flag).to(device)
    weight = ref_model.conv_transpose3d.weight.detach().clone()
    bias_tensor = None

    custom_kernel = build_kernel()

    out_custom = custom_kernel.forward(
        x, weight, bias_tensor,
        [stride[0], stride[1], stride[2]],
        [padding[0], padding[1], padding[2]],
        [output_padding[0], output_padding[1], output_padding[2]],
        groups
    )
    out_ref = ref_model(x)
    # Even if the block size is not optimal for these dimensions, correctness should remain.
    # However, due to potential hard-coded block size issues, the results may be off.
    diff = (out_custom - out_ref).abs().max().item()
    assert diff > 1e-3, f"Expected difference due to non-generalized block size handling, but diff={diff}"
