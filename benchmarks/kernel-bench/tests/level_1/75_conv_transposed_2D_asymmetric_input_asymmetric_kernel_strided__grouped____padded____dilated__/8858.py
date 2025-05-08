
import torch
import pytest
from torch import nn
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_conv_transpose2d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Helper function to run a native ConvTranspose2d for reference.
def native_conv_transpose2d(x, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias_flag):
    conv = nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias_flag
    ).to(x.device)
    # Initialize weights and bias to match the kernel expectation.
    # Our custom kernel expects weight shape [in_channels, out_channels_per_group, kernel_h, kernel_w]
    with torch.no_grad():
        conv.weight.copy_(torch.randn_like(conv.weight))
        if bias_flag:
            conv.bias.copy_(torch.randn_like(conv.bias))
    return conv(x), conv.weight, (conv.bias if bias_flag else None)

# Test 1: Trigger issue with candidate offset
# Use stride and dilation where gcd(stride, dilation) > 1 so that only a subset of mod residues are generated.
# For example, with stride=4 and dilation=2 → valid residues are only 0 and 2.
# Choose padding such that candidate + pad gives a mod value not in {0,2} (e.g. 1).
def test_candidate_offset_issue():
    # Use small spatial size to make candidate_h = oh + pad small.
    batch_size = 1
    in_channels = 4
    out_channels = 4
    kernel_size = (3, 3)
    stride = (4, 4)           # For both dims
    dilation = (2, 2)         # gcd(4,2)=2 → only 0 and 2 are produced
    padding = (1, 1)          # so for oh=0, candidate_h = 1, and 1 mod 4 == 1 (not in {0,2})
    groups = 1
    bias = True

    # Input dimensions chosen to provoke the problematic candidate index computation.
    height, width = 5, 5
    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)
    
    # Build the native conv-transpose2d for reference.
    ref_output, weight, ref_bias = native_conv_transpose2d(
        x, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
    )
    
    # Build the custom kernel’s module.
    mod = build_kernel()
    # The custom kernel expects:
    # x, weight, bias(optional), stride, padding, dilation, groups
    # Ensure weight and bias exactly match the native ones.
    if ref_bias is None:
        bias_tensor = torch.empty(0, device="cuda", dtype=torch.float32)
    else:
        bias_tensor = ref_bias
    custom_out = mod.forward(
        x, weight, bias_tensor, list(stride), list(padding), list(dilation), groups
    )
    torch.cuda.synchronize()

    # If the candidate offset issue occurs, the custom kernel output will deviate from the reference.
    err = (custom_out - ref_output).abs().max().item()
    assert err > 1e-3, f"Test candidate offset issue: Expected discrepancy due to offset issue, but error={err}"


# Test 2: Trigger issue with improper loop unrolling on dynamic loop bounds.
# Use parameters which produce dynamic loop bounds (by making output dimensions or kernel bounds not compile-time constants).
def test_loop_unrolling_dynamic_bounds():
    batch_size = 2
    in_channels = 8
    out_channels = 8
    kernel_size = (5, 7)      # non-square, larger kernels make loops more dynamic
    stride = (2, 3)
    dilation = (1, 2)
    padding = (2, 3)
    groups = 1
    bias = True

    height, width = 10, 12
    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)

    ref_output, weight, ref_bias = native_conv_transpose2d(
        x, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
    )

    mod = build_kernel()
    if ref_bias is None:
        bias_tensor = torch.empty(0, device="cuda", dtype=torch.float32)
    else:
        bias_tensor = ref_bias
    custom_out = mod.forward(
        x, weight, bias_tensor, list(stride), list(padding), list(dilation), groups
    )
    torch.cuda.synchronize()

    # Due to loop unrolling issues the result may deviate.
    err = (custom_out - ref_output).abs().max().item()
    assert err > 1e-3, f"Test loop unrolling dynamic bounds: Expected discrepancy due to unrolling dynamic loops, but error={err}"


# Test 3: Trigger issue with strict divisibility assumption in computing input indices.
# Choose parameters such that (candidate - k*dilation) is not divisible by stride.
def test_divisibility_assumption_issue():
    batch_size = 1
    in_channels = 4
    out_channels = 4
    kernel_size = (3, 3)
    stride = (2, 2)
    dilation = (2, 2)  # With these values, if padding is chosen oddly, divisibility may fail.
    padding = (1, 1)   # For oh=0 -> candidate_h=1 which is not divisible by 2.
    groups = 1
    bias = True

    height, width = 6, 6
    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)

    ref_output, weight, ref_bias = native_conv_transpose2d(
        x, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
    )

    mod = build_kernel()
    if ref_bias is None:
        bias_tensor = torch.empty(0, device="cuda", dtype=torch.float32)
    else:
        bias_tensor = ref_bias
    custom_out = mod.forward(
        x, weight, bias_tensor, list(stride), list(padding), list(dilation), groups
    )
    torch.cuda.synchronize()

    err = (custom_out - ref_output).abs().max().item()
    assert err > 1e-3, f"Test divisibility assumption issue: Expected discrepancy due to strict divisibility assumption, but error={err}"


# Test 4: Trigger issue with improper handling of groups when in_channels is not divisible by groups.
def test_group_division_issue():
    # in_channels not divisible by groups.
    batch_size = 1
    in_channels = 10  # deliberately not divisible by groups=3
    out_channels = 6  # out_channels per group will be computed as (6/3 = 2) but one input channel is extra.
    kernel_size = (3, 3)
    stride = (1, 1)
    dilation = (1, 1)
    padding = (1, 1)
    groups = 3
    bias = True

    height, width = 8, 8
    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32)

    # PyTorch's native ConvTranspose2d will raise an error if channels are incompatible.
    with pytest.raises(Exception):
        native_conv_transpose2d(x, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    # Our custom kernel does no such validation, so it will run:
    mod = build_kernel()
    # Create dummy weight with shape: [in_channels, out_channels_per_group, kernel_h, kernel_w]
    # Here out_channels_per_group = out_channels / groups = 2.
    weight = torch.randn(in_channels, out_channels // groups, kernel_size[0], kernel_size[1], device="cuda", dtype=torch.float32)
    bias_tensor = torch.randn(out_channels, device="cuda", dtype=torch.float32) if bias else torch.empty(0, device="cuda", dtype=torch.float32)
    # It might produce an output though processing only part of the channels.
    custom_out = mod.forward(
        x, weight, bias_tensor, list(stride), list(padding), list(dilation), groups
    )
    torch.cuda.synchronize()
    # We simply check that the output is not equal to some expected proper behavior;
    # in a real scenario we would want the kernel to validate inputs.
    assert custom_out is not None, "Custom kernel did not produce an output despite invalid group division."


# Test 5: Trigger issue due to lack of device synchronization.
# Pass a CPU tensor to force the kernel launch to fail asynchronously.
def test_no_device_synchronization_issue():
    batch_size = 1
    in_channels = 4
    out_channels = 4
    kernel_size = (3, 3)
    stride = (1, 1)
    dilation = (1, 1)
    padding = (1, 1)
    groups = 1
    bias = True

    height, width = 8, 8
    x = torch.randn(batch_size, in_channels, height, width, dtype=torch.float32)  # CPU tensor

    mod = build_kernel()
    weight = torch.randn(in_channels, out_channels // groups, kernel_size[0], kernel_size[1], dtype=torch.float32)
    bias_tensor = torch.randn(out_channels, dtype=torch.float32) if bias else torch.empty(0, dtype=torch.float32)
    with pytest.raises(RuntimeError):
        # This should error because the kernel expects CUDA tensors.
        mod.forward(x, weight, bias_tensor, list(stride), list(padding), list(dilation), groups)
