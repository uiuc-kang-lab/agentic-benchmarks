
import os
import pytest
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

# Utility function to compile and load the CUDA kernel extension.
def build_kernel(extra_cuda_cflags=None):
    extra_cuda_cflags = extra_cuda_cflags or ["-O3", "--use_fast_math"]
    cuda_module = load(
        name="custom_transposed_conv3d",
        sources=["kernel.cu"],
        extra_cuda_cflags=extra_cuda_cflags,
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

#
# Test 1: Shared Memory Allocation Type Issue
#
def test_shared_memory_allocation_for_double():
    # Create a transposed conv with double precision.
    batch_size = 2
    in_channels = 4
    out_channels = 3
    kernel_size = (3, 3, 3)
    stride = (1, 1, 1)
    padding = (1, 1, 1)
    output_padding = (0, 0, 0)
    groups = 1
    bias = True

    # Use nn.ConvTranspose3d as reference with dtype torch.float64.
    ref_conv = nn.ConvTranspose3d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding,
        output_padding=output_padding, groups=groups,
        bias=bias,
    ).double().cuda()

    # Get input in double
    input_tensor = torch.randn(batch_size, in_channels, 8, 8, 8, dtype=torch.float64, device="cuda")

    # Run reference forward
    ref_output = ref_conv(input_tensor)

    # Build our kernel extension
    mod = build_kernel()

    # Mimic weight layout for our kernel:
    # Expected weight shape for our kernel: [in_channels, out_channels/groups, kT, kH, kW]
    weight = ref_conv.weight.detach().clone()  # Already in double
    bias_tensor = ref_conv.bias.detach().clone() if bias else None

    # Call our custom forward.
    # This call will use double type, but the shared memory allocation in the kernel is computed with sizeof(float),
    # so the output is likely wrong.
    output = mod.forward(
        input_tensor,
        weight,
        bias_tensor,
        list(stride),
        list(padding),
        list(output_padding),
        groups
    )
    torch.cuda.synchronize()

    # Check that the outputs are (incorrectly) different.
    # We expect the error to be triggered by the wrong shared memory allocation.
    assert not torch.allclose(output, ref_output, atol=1e-5), \
        "Kernel output should differ from reference when using double precision due to shared memory size miscalculation."

#
# Test 2: Output Padding Handling Issue
#
def test_output_padding_handling():
    # Set non-zero output_padding.
    batch_size = 2
    in_channels = 4
    out_channels = 3
    kernel_size = (3, 3, 3)
    stride = (2, 2, 2)
    padding = (1, 1, 1)
    output_padding = (1, 1, 1)  # Non-zero output padding.
    groups = 1
    bias = False

    ref_conv = nn.ConvTranspose3d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding,
        output_padding=output_padding, groups=groups,
        bias=bias,
    ).cuda()

    input_tensor = torch.randn(batch_size, in_channels, 5, 5, 5, device="cuda", dtype=torch.float32)

    ref_output = ref_conv(input_tensor)

    mod = build_kernel()

    weight = ref_conv.weight.detach().clone()
    bias_tensor = None

    # Our custom kernel does compute the output dimensions in the host function; however,
    # it does NOT incorporate output_padding in the coordinate calculation inside the kernel.
    # This test should show a difference between the custom kernel and the reference.
    output = mod.forward(
        input_tensor,
        weight,
        bias_tensor,
        list(stride),
        list(padding),
        list(output_padding),
        groups
    )
    torch.cuda.synchronize()

    # The outputs should differ because of the missing output_padding handling.
    assert not torch.allclose(output, ref_output, atol=1e-5), \
        "Kernel output unexpectedly matches reference output. Output_padding handling appears to be implemented, but it should not be."

#
# Test 3: Excessive Grid Size Issue
#
def test_excessive_grid_size():
    # Simulate a scenario where output dimensions are so large that the number of blocks
    # (one block per output element) exceeds CUDA's capability.
    # We create a small input but set stride and kernel such that the computed output is enormous.
    batch_size = 1
    in_channels = 1
    out_channels = 1
    kernel_size = (3, 3, 3)
    # Choose stride, padding and output_padding to artificially inflate output dimensions.
    stride = (10, 10, 10)
    padding = (0, 0, 0)
    output_padding = (0, 0, 0)
    groups = 1
    bias = False

    # Set input to a size that makes output huge.
    # For example, input depth, height, width = 5 produces:
    # out_depth = (5 - 1) * 10 + 3 = 43 (similarly for height and width)
    # Although 43^3 is not huge, we simulate an excessive grid by manipulating input dimensions.
    # Here, we deliberately set a large batch size to inflate total output elements.
    batch_size = 100000  # large batch size to force an enormous grid dimension

    ref_conv = nn.ConvTranspose3d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding,
        output_padding=output_padding, groups=groups,
        bias=bias,
    ).cuda()

    input_tensor = torch.randn(batch_size, in_channels, 5, 5, 5, device="cuda", dtype=torch.float32)
    weight = ref_conv.weight.detach().clone()
    bias_tensor = None

    mod = build_kernel()
    # Expect the kernel launch to fail due to too many blocks.
    with pytest.raises(RuntimeError):
        output = mod.forward(
            input_tensor,
            weight,
            bias_tensor,
            list(stride),
            list(padding),
            list(output_padding),
            groups
        )
        torch.cuda.synchronize()

#
# Test 4: Misleading Comment on Warp-level Primitives
#
def test_warp_level_comment_consistency():
    # Read the source file to check whether __shfl_down_sync is actually used in the code.
    # The comment claims its usage but the implementation uses shared memory reduction instead.
    kernel_source = ""
    kernel_filename = os.path.join(os.path.dirname(__file__), "kernel.cu")
    with open(kernel_filename, "r") as f:
        kernel_source = f.read()
    # Check if __shfl_down_sync is mentioned in the code.
    has_shfl_down_sync = "__shfl_down_sync" in kernel_source
    # The comment is misleading if it says it's used but it doesn't appear in the code.
    assert not has_shfl_down_sync, "Kernel source mentions __shfl_down_sync, but the implementation does not use it as described in the comments."

#
# Test 5: Lack of Post-Kernel-Launch Error Checking
#
def test_no_post_launch_error_checking():
    # We can simulate an erroneous launch by deliberately providing invalid tensor shapes.
    # The expectation is that without proper error checking inside the kernel,
    # the error will only be caught much later (or give an unclear error message).
    batch_size = 2
    in_channels = 4
    out_channels = 3
    kernel_size = (3, 3, 3)
    stride = (1, 1, 1)
    padding = (1, 1, 1)
    output_padding = (0, 0, 0)
    groups = 1
    bias = False

    ref_conv = nn.ConvTranspose3d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding,
        output_padding=output_padding, groups=groups,
        bias=bias,
    ).cuda()

    # Intentionally make input tensor with an incorrect number of dimensions (e.g. missing one spatial dimension)
    # which should produce an error in kernel execution.
    input_tensor = torch.randn(batch_size, in_channels, 8, 8, device="cuda", dtype=torch.float32)
    weight = ref_conv.weight.detach().clone()
    bias_tensor = None

    mod = build_kernel()
    with pytest.raises(RuntimeError):
        output = mod.forward(
            input_tensor,
            weight,
            bias_tensor,
            list(stride),
            list(padding),
            list(output_padding),
            groups
        )
        torch.cuda.synchronize()
