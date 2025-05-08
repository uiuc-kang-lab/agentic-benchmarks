
import pytest
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os

# Build the CUDA kernel module
def build_kernel():
    # Ensure that the extension is rebuilt each time this test is run.
    module = load(
        name="kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return module

# Helper function to run the custom forward
def run_custom_conv_transpose2d(module, x, weight, bias, stride, padding, dilation, groups):
    return module.forward(x, weight, bias, stride, padding, dilation, groups)

# Test 1: Test kernel with double precision input (should trigger the float32-only issue)
def test_dtype_issue():
    module = build_kernel()
    batch_size = 4
    in_channels = 8
    out_channels = 8
    kernel_size = (3, 3)
    stride = [1, 1]
    padding = [1, 1]
    dilation = [1, 1]
    groups = 1

    # Create input and weight in double precision.
    x = torch.randn(batch_size, in_channels, 16, 16, device="cuda", dtype=torch.double)
    # nn.ConvTranspose2d weight shape: [in_channels, out_channels/groups, kH, kW]
    weight = torch.randn(in_channels, out_channels // groups, kernel_size[0], kernel_size[1],
                         device="cuda", dtype=torch.double)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.double)

    # Call the custom kernel. It expects float32.
    with pytest.raises(RuntimeError):
        # This should raise an error or produce wrong results due to dtype mismatch.
        out = run_custom_conv_transpose2d(module, x, weight, bias, stride, padding, dilation, groups)
        torch.cuda.synchronize()

# Test 2: Test kernel with weight layout/grouping issue.
# Although the forward() makes the weights contiguous, if the kernel is used in a more general setting,
# the assumption about the memory layout might break.
def test_weight_layout_issue():
    module = build_kernel()
    batch_size = 2
    in_channels = 8
    out_channels = 8
    kernel_size = (3, 5)
    stride = [2, 3]
    padding = [1, 2]
    dilation = [2, 1]
    groups = 2

    # Create inputs as in ConvTranspose2d.
    x = torch.randn(batch_size, in_channels, 24, 24, device="cuda", dtype=torch.float32)
    # Create weight with deliberate non-standard layout.
    weight = torch.randn(in_channels, out_channels // groups, kernel_size[0], kernel_size[1],
                         device="cuda", dtype=torch.float32)
    # Make weight non-contiguous by transposing two dimensions.
    weight = weight.transpose(1, 2)
    # Do not call contiguous() so that if the kernel were to be used without enforcing contiguous memory,
    # the layout assumption might fail. (The current forward() does call contiguous(), but in a more general situation it might not.)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    # Call custom kernel. We compare against PyTorch's ConvTranspose2d using contiguous weight.
    custom_out = run_custom_conv_transpose2d(module, x, weight, bias, stride, padding, dilation, groups)
    torch.cuda.synchronize()
    # Build reference op with contiguous weight
    conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=True).cuda()
    # Inject the same non-contiguous weight (converted to contiguous for reference) and bias:
    conv.weight.data = weight.contiguous()
    conv.bias.data = bias
    ref_out = conv(x)
    # The outputs are likely to differ if the kernel relies on a specific layout assumption.
    assert not torch.allclose(custom_out, ref_out, atol=1e-4), \
        "Kernel output unexpectedly matches reference output despite non-standard weight layout."

# Test 3: Test kernel with in_channels not divisible by groups.
def test_group_divisibility_issue():
    module = build_kernel()
    batch_size = 2
    in_channels = 10  # Not divisible by groups=3.
    out_channels = 6
    kernel_size = (3, 3)
    stride = [1, 1]
    padding = [1, 1]
    dilation = [1, 1]
    groups = 3

    x = torch.randn(batch_size, in_channels, 16, 16, device="cuda", dtype=torch.float32)
    # For ConvTranspose2d, weight shape is [in_channels, out_channels/groups, kH, kW].
    weight = torch.randn(in_channels, out_channels // groups, kernel_size[0], kernel_size[1],
                         device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    # The kernel itself does not check for divisibility.
    # We compare the output with PyTorch's ConvTranspose2d to see if the error manifests.
    custom_out = run_custom_conv_transpose2d(module, x, weight, bias, stride, padding, dilation, groups)
    torch.cuda.synchronize()
    conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=True).cuda()
    conv.weight.data = weight
    conv.bias.data = bias
    ref_out = conv(x)

    # The outputs will likely differ due to wrong indexing caused by incorrect division.
    assert not torch.allclose(custom_out, ref_out, atol=1e-4), \
        "Kernel output matches reference output even though in_channels is not divisible by groups, which is unexpected."

# Test 4: Test kernel with excessive shared memory usage.
# We construct a case where the weight tensor is huge such that the required shared memory
# might exceed device limits, triggering a kernel launch error (the kernel prints an error message).
def test_shared_memory_issue():
    module = build_kernel()
    # Construct parameters that force a large shared memory requirement.
    batch_size = 1
    in_channels = 256  # high number of channels
    out_channels = 256
    kernel_size = (7, 7)
    stride = [1, 1]
    padding = [3, 3]
    dilation = [1, 1]
    groups = 1

    x = torch.randn(batch_size, in_channels, 32, 32, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels // groups, kernel_size[0], kernel_size[1],
                         device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    # Launch the kernel. If the shared memory allocation exceeds device limits,
    # the kernel launch will fail and the printed error message (from cudaGetLastError)
    # should indicate an error.
    custom_out = run_custom_conv_transpose2d(module, x, weight, bias, stride, padding, dilation, groups)
    torch.cuda.synchronize()
    # Since the kernel does not raise an exception even on launch failure (only a print),
    # we perform a simple check on the output size.
    # In a real situation, one would parse the log to ensure the error was triggered.
    assert custom_out.numel() == batch_size * out_channels * 32 * 32, \
        "Kernel output size does not match expected dimensions which might indicate a shared memory allocation issue."

if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])
