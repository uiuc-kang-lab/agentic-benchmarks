
import torch
import pytest
from torch import nn
from torch.utils.cpp_extension import load

# Build the CUDA extension from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="custom_conv_transpose2d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# A helper to run the custom kernel.
def run_custom_conv_transpose2d(input, convtrans, kernel_module):
    # Get the weight and bias from the provided module.
    weight = convtrans.weight
    bias = convtrans.bias if convtrans.bias is not None else torch.tensor([], device=input.device)
    stride = convtrans.stride[0]
    padding = convtrans.padding[0]
    dilation = convtrans.dilation[0]
    # Call the custom op (wrapper name "forward" as defined in the module)
    return kernel_module.forward(input, weight, bias, stride, padding, dilation)

# Test 1: Groups are not supported.
# In PyTorch, a ConvTranspose2d layer created with groups != 1 will yield a correct result,
# but our custom kernel ignores groups so the output will differ.
def test_groups():
    kernel_module = build_kernel()
    batch_size = 2
    in_channels = 4
    out_channels = 4
    kernel_size = 3
    groups = 2  # groups != 1; valid if in_channels % groups == 0 and out_channels % groups == 0.
    # Create input
    x = torch.randn(batch_size, in_channels, 16, 16, device="cuda", dtype=torch.float32)
    # Build a PyTorch layer with groups. (Our custom kernel does not support groups.)
    convtrans = nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size,
        stride=1, padding=1, dilation=1, groups=groups, bias=True
    ).cuda()
    # Get reference (from PyTorch)
    ref = convtrans(x)
    # Get custom kernel result.
    custom_out = run_custom_conv_transpose2d(x, convtrans, kernel_module)
    # They should not match because groups are not handled.
    assert not torch.allclose(custom_out, ref, atol=1e-4), \
        f"Test groups expected a mismatch due to unsupported groups, but got matching outputs."

# Test 2: Double buffering issues when processing many output chunks.
# We force a large output so that the chunking (with double buffering) is heavily exercised.
def test_double_buffering():
    kernel_module = build_kernel()
    batch_size = 2
    in_channels = 8
    out_channels = 8
    kernel_size = 3
    stride = 8   # Large stride makes output much larger than input.
    padding = 1
    dilation = 1
    # Create relatively small input
    x = torch.randn(batch_size, in_channels, 8, 8, device="cuda", dtype=torch.float32)
    # PyTorch layer: note groups defaults to 1.
    convtrans = nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding, dilation=dilation, bias=True
    ).cuda()
    ref = convtrans(x)
    custom_out = run_custom_conv_transpose2d(x, convtrans, kernel_module)
    # Expect a significant mismatch because the kernel’s double buffering logic does not really overlap work correctly.
    assert not torch.allclose(custom_out, ref, atol=1e-4), \
        f"Test double buffering expected a mismatch, but outputs are similar."

# Test 3: Kernel unrolling issue with dynamic (non-constant) kernel_size.
# Using a kernel_size that is not known at compile time (e.g. 5 instead of 3).
def test_dynamic_kernel_size():
    kernel_module = build_kernel()
    batch_size = 2
    in_channels = 4
    out_channels = 6
    kernel_size = 5   # Non-standard, dynamic kernel size.
    stride = 2
    padding = 2
    dilation = 1
    x = torch.randn(batch_size, in_channels, 16, 16, device="cuda", dtype=torch.float32)
    convtrans = nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding, dilation=dilation, bias=True
    ).cuda()
    ref = convtrans(x)
    custom_out = run_custom_conv_transpose2d(x, convtrans, kernel_module)
    # Expect mismatch because the loop unrolling may not work correctly for a dynamic kernel size.
    assert not torch.allclose(custom_out, ref, atol=1e-4), \
        f"Test dynamic kernel size expected a mismatch, but outputs are similar."

# Test 4: Kernel does not check or support non-float32 (e.g. float64) inputs.
# Passing a float64 tensor should produce wrong results.
def test_dtype():
    kernel_module = build_kernel()
    batch_size = 2
    in_channels = 4
    out_channels = 4
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    # Create input with float64 instead of float32.
    x = torch.randn(batch_size, in_channels, 16, 16, device="cuda", dtype=torch.float64)
    convtrans = nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding, dilation=dilation, bias=True
    ).cuda()
    ref = convtrans(x.to(torch.float32)).to(torch.float64)  # reference computed in float32 then cast
    custom_out = run_custom_conv_transpose2d(x.to(torch.float32), convtrans, kernel_module).to(torch.float64)
    # Although we cast, the custom kernel was built for float32 and using float64 originally is not supported. 
    # Here we simulate the issue: if one calls with float64, it should be an error or mismatch.
    # We force a mismatch check.
    assert not torch.allclose(custom_out, ref, atol=1e-4), \
        f"Test dtype expected a mismatch due to unsupported float64, but outputs are similar."

# Test 5: Kernel assumes contiguous tensors.
# Passing a non-contiguous tensor should result in an incorrect result.
def test_non_contiguous():
    kernel_module = build_kernel()
    batch_size = 2
    in_channels = 4
    out_channels = 4
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    x = torch.randn(batch_size, in_channels, 16, 16, device="cuda", dtype=torch.float32)
    # Make x non-contiguous by transposing a spatial dimension.
    x_noncontig = x.transpose(2, 3)
    # Build the layer – note that weight from convtrans will be contiguous.
    convtrans = nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding, dilation=dilation, bias=True
    ).cuda()
    ref = convtrans(x_noncontig.contiguous())  # reference uses contiguous version.
    custom_out = run_custom_conv_transpose2d(x_noncontig, convtrans, kernel_module)
    # Expect a mismatch because the kernel does not explicitly handle non-contiguous memory.
    assert not torch.allclose(custom_out, ref, atol=1e-4), \
        f"Test non-contiguous expected a mismatch due to non-contiguous input, but outputs are similar."

if __name__ == "__main__":
    pytest.main([__file__])
