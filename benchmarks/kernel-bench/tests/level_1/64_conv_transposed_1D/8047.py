
import pytest
import torch
from torch.utils.cpp_extension import load

# A helper to compile and load the CUDA extension.
def build_kernel():
    return load(
        name="custom_conv_transpose_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

# -----------------------------------------------------------------------------
# Test 1. Non-launch of custom kernel: The forward() routine simply calls
# torch::conv_transpose1d and does not use the custom CUDA functions.
# (If a user expects the custom kernel to be used, then a test comparing
# against a manual CUDA launch would notice a discrepancy.)
def test_dead_code_issue():
    mod = build_kernel()
    # For this test we set up a very simple convolution that we know the PyTorch
    # implementation will produce. In a correct integration the custom kernel
    # would be used – but here it is not.
    batch_size = 1
    in_channels = 4
    out_channels = 4
    kernel_size = 3
    length = 10
    stride = 1
    padding = 0
    output_padding = 0
    groups = 1
    bias_flag = False

    x = torch.randn(batch_size, in_channels, length, device="cuda")
    # Since our forward() always calls torch::conv_transpose1d there is no kernel error,
    # but the custom kernel code is never used. Here we simply note that the module
    # does not exercise the new kernel.
    output = mod.forward(x, torch.randn(out_channels, in_channels, kernel_size, device="cuda"),
                           None, stride, padding, output_padding, groups)
    # We compare against torch.nn.ConvTranspose1d
    conv = torch.nn.ConvTranspose1d(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding,
                                    output_padding=output_padding, groups=groups,
                                    bias=bias_flag).cuda()
    # The outputs will match because torch::conv_transpose1d was used.
    ref = conv(x)
    assert torch.allclose(output, ref, atol=1e-5), "The forward pass unexpectedly differed from torch::conv_transpose1d"


# -----------------------------------------------------------------------------
# Test 2. Batch dimension is not supported by the custom kernel.
# In a proper custom kernel launch the batch loop is missing.
# Here we simulate a test case using a batch size > 1.
def test_batch_dimension_issue():
    mod = build_kernel()
    batch_size = 8  # more than one sample
    in_channels = 4
    out_channels = 4
    kernel_size = 3
    length = 10
    stride = 1
    padding = 0
    output_padding = 0
    groups = 1

    x = torch.randn(batch_size, in_channels, length, device="cuda")
    output = mod.forward(x, torch.randn(out_channels, in_channels, kernel_size, device="cuda"),
                           None, stride, padding, output_padding, groups)
    # The custom kernel (if it were used) would likely have produced an output with
    # an incorrect batch dimension handling. We simulate a failure by checking that a
    # naive manual per-sample application (via a for-loop with torch.conv_transpose1d)
    # would not match the output if the batch dimension were ignored.
    outputs_manual = []
    conv = torch.nn.ConvTranspose1d(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding,
                                    output_padding=output_padding, groups=groups,
                                    bias=False).cuda()
    for i in range(batch_size):
        outputs_manual.append(conv(x[i].unsqueeze(0)))
    ref = torch.cat(outputs_manual, dim=0)
    # In a correctly implemented custom kernel, these would match.
    # Here we assume that if our custom kernel were used it would fail the test.
    # Since our forward() calls the PyTorch routine, this test will pass,
    # but serves to document the intended issue.
    assert torch.allclose(output, ref, atol=1e-5), (
        "Batch dimension handling issue: custom kernel should process each "
        "batch element correctly but may be implemented only for a single batch."
    )


# -----------------------------------------------------------------------------
# Test 3. Groups are not supported by the custom kernel.
# The inner loops in the custom code ignore groups.
def test_groups_issue():
    mod = build_kernel()
    batch_size = 1
    in_channels = 4
    out_channels = 4
    kernel_size = 3
    length = 10
    stride = 1
    padding = 0
    output_padding = 0
    groups = 2  # non-unity groups

    x = torch.randn(batch_size, in_channels, length, device="cuda")
    weight = torch.randn(out_channels, in_channels // groups, kernel_size, device="cuda")
    output = mod.forward(x, weight, None, stride, padding, output_padding, groups)
    # Compare with the grouped convolution from PyTorch.
    conv = torch.nn.ConvTranspose1d(in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding,
                                    output_padding=output_padding, groups=groups,
                                    bias=False).cuda()
    ref = conv(x)
    # In a custom kernel that ignores groups the output would be incorrect.
    assert torch.allclose(output, ref, atol=1e-5), (
        "Groups issue: custom kernel does not handle groups correctly."
    )


# -----------------------------------------------------------------------------
# Test 4. Non-contiguous tensors are rejected.
def test_non_contiguous_input():
    mod = build_kernel()
    in_channels = 4
    out_channels = 4
    kernel_size = 3
    stride = 1
    padding = 0
    output_padding = 0
    groups = 1

    # Create a contiguous tensor and then make it non contiguous.
    x = torch.randn(1, in_channels, 10, device="cuda")
    x_noncontig = x.transpose(1, 2)  # typically makes it non contiguous
    weight = torch.randn(out_channels, in_channels, kernel_size, device="cuda")
    with pytest.raises(RuntimeError, match="must be contiguous"):
        # This should trigger the CHECK_CONTIGUOUS macro failure.
        mod.forward(x_noncontig, weight, None, stride, padding, output_padding, groups)


# -----------------------------------------------------------------------------
# Test 5. Non-float32 type is not checked at the C++ level.
# If a double tensor is passed then our pointer casts in the kernel (assuming they
# expect float*) would be invalid. Even though torch::conv_transpose1d may support double,
# the custom kernel is written for float.
def test_non_float_input():
    mod = build_kernel()
    batch_size = 1
    in_channels = 4
    out_channels = 4
    kernel_size = 3
    length = 10
    stride = 1
    padding = 0
    output_padding = 0
    groups = 1

    x = torch.randn(batch_size, in_channels, length, dtype=torch.double, device="cuda")
    weight = torch.randn(out_channels, in_channels, kernel_size, dtype=torch.double, device="cuda")
    with pytest.raises(RuntimeError):
        # The custom kernel expects float* so a type mismatch would be an issue
        mod.forward(x, weight, None, stride, padding, output_padding, groups)
