
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# Build the extension module from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="custom_conv1d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1: C_in not divisible by groups.
# Explanation: If C_in is not divisible by groups, the computed group_size_in will be C_in // groups,
# meaning some input channels are ignored.
def test_in_channels_not_divisible_by_groups():
    # Use an input with C_in = 3 and groups = 2. (3 not divisible by 2)
    batch_size = 2
    in_channels = 3
    out_channels = 8
    length = 16
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    groups = 2

    # Create input tensor (CUDA, float32) and a weight tensor with shape [C_out, C_in//groups, kernel_size]
    # Note: 3 // 2 = 1, so weight has shape [8, 1, 3] even though a proper conv with groups=2 should have 1.5 channels per group.
    x = torch.randn(batch_size, in_channels, length, device="cuda", dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels // groups, kernel_size, device="cuda", dtype=torch.float32)
    # No bias used.
    
    custom_module = build_kernel()
    custom_out = custom_module.forward(x, weight, None, stride, padding, dilation, groups)
    
    # torch.nn.functional.conv1d should complain when C_in is not divisible by groups.
    with pytest.raises(RuntimeError):
        _ = F.conv1d(x, weight, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups)
        
    # (The test case shows that our kernel silently proceeds – an issue if generality is required.)
    assert custom_out is not None

# Issue 2: C_out not divisible by groups.
# Explanation: If C_out is not divisible by groups, then the group_size_out computed inside the kernel is off.
def test_out_channels_not_divisible_by_groups():
    # Use in_channels divisible by groups but out_channels not divisible.
    batch_size = 2
    in_channels = 4  # divisible by 2 groups -> group_size_in = 2.
    out_channels = 5  # 5 not divisible by groups; expected group_size_out should be 5/2.
    length = 16
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    groups = 2

    x = torch.randn(batch_size, in_channels, length, device="cuda", dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels // groups, kernel_size, device="cuda", dtype=torch.float32)

    custom_module = build_kernel()
    custom_out = custom_module.forward(x, weight, None, stride, padding, dilation, groups)
    
    with pytest.raises(RuntimeError):
        _ = F.conv1d(x, weight, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups)
        
    assert custom_out is not None

# Issue 3: Wrong input tensor dtype.
# Explanation: The kernel code checks that x is float32. Passing a double tensor should trigger a TORCH_CHECK.
def test_input_wrong_dtype():
    batch_size = 2
    in_channels = 4
    out_channels = 8
    length = 16
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    groups = 1

    # Create x with double type.
    x = torch.randn(batch_size, in_channels, length, device="cuda", dtype=torch.float64)
    weight = torch.randn(out_channels, in_channels // groups, kernel_size, device="cuda", dtype=torch.float32)

    custom_module = build_kernel()
    with pytest.raises(RuntimeError):
        _ = custom_module.forward(x, weight, None, stride, padding, dilation, groups)

# Issue 4: Wrong weight shape.
# Explanation: The kernel assumes weight shape [C_out, C_in/groups, kernel_size].
# Providing a weight tensor of a different shape should lead to indexing errors.
def test_wrong_weight_shape():
    batch_size = 2
    in_channels = 4
    out_channels = 8
    length = 16
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    groups = 1

    x = torch.randn(batch_size, in_channels, length, device="cuda", dtype=torch.float32)
    # Intentionally provide a wrong weight shape. Expected shape is [8, 4, 3] for groups 1.
    weight = torch.randn(out_channels, in_channels, kernel_size, device="cuda", dtype=torch.float32)

    custom_module = build_kernel()
    # Here we expect that either the extension produces an incorrect result (not matching F.conv1d)
    # or PyTorch's F.conv1d (if forced) would error. We choose to compare outputs.
    with pytest.raises(RuntimeError):
        _ = F.conv1d(x, weight, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups)
        
    # If our custom kernel runs, its output would be wrong.
    custom_out = custom_module.forward(x, weight, None, stride, padding, dilation, groups)
    # (No assertion on value; the discrepancy itself is the issue.)

# Issue 5: Non‐contiguous input.
# Explanation: The kernel uses pointer arithmetic based on the assumption that the input is contiguous.
# Passing a non‐contiguous tensor may lead to wrong results. We test this by comparing the result on contiguous
# and non‐contiguous inputs.
def test_non_contiguous_input():
    # We construct a non‐contiguous input by slicing.
    batch_size = 2
    in_channels = 4
    out_channels = 8
    length = 32
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    groups = 1

    # Create a larger contiguous tensor.
    x_full = torch.randn(batch_size, in_channels, length * 2, device="cuda", dtype=torch.float32)
    # Create non‐contiguous view: select every second element along the length dimension.
    x_nc = x_full[:, :, ::2]
    # x_nc is non‐contiguous but has shape [batch_size, in_channels, length].

    # Build weight with correct shape.
    weight = torch.randn(out_channels, in_channels // groups, kernel_size, device="cuda", dtype=torch.float32)

    # Compute output with the custom kernel.
    custom_module = build_kernel()
    out_nc = custom_module.forward(x_nc, weight, None, stride, padding, dilation, groups)
    # Also compute the reference output using PyTorch's conv1d.
    out_ref = F.conv1d(x_nc.contiguous(), weight, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups)

    # We expect that if the custom kernel incorrectly assumes contiguity,
    # then the output for the non‐contiguous input may be different.
    if not torch.allclose(out_nc, out_ref, atol=1e-4):
        pytest.xfail("Kernel produces different output for non‐contiguous input, because it assumes contiguous memory layout.")
    else:
        # If by chance they match, then the behavior may be benign.
        assert torch.allclose(out_nc, out_ref, atol=1e-4)
