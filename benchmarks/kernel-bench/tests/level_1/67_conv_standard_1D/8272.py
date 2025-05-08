
import torch
import pytest
from torch.utils.cpp_extension import load
import torch.nn as nn

# Helper function to build and load the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="custom_conv1d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# 1. Test for kernel width overflow when kernel_size > 32.
def test_kernel_size_overflow():
    # Create input where kernel_size is larger than 32
    batch_size = 4
    in_channels = 3
    out_channels = 8
    kernel_size = 33  # > 32, will overflow weight_cache
    length = 64
    stride = 1
    padding = 0
    dilation = 1
    groups = 1
    bias = False

    # Use PyTorch's Conv1d for reference
    ref_conv = nn.Conv1d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias
    ).cuda()
    # Force weight and bias to float32
    ref_conv.weight.data = torch.randn_like(ref_conv.weight.data).float()
    if bias:
        ref_conv.bias.data = torch.randn_like(ref_conv.bias.data).float()

    # Create input tensor
    x = torch.randn(batch_size, in_channels, length, device="cuda", dtype=torch.float32)
    ref_y = ref_conv(x)

    # Build and call our custom kernel
    mod = build_kernel()
    # Our custom kernel forward expects: x, weight, bias, stride, padding, dilation, groups.
    # Pass bias as None if not provided.
    custom_y = mod.forward(
        x,
        ref_conv.weight,
        None if not bias else ref_conv.bias,
        stride,
        padding,
        dilation,
        groups,
    )
    torch.cuda.synchronize()
    # In the overflow scenario, the result will likely differ.
    assert not torch.allclose(custom_y, ref_y, atol=1e-4), "When kernel_size > 32, the custom kernel should fail due to weight_cache overflow."

# 2. Test for non-16-byte aligned tensors.
def test_non_aligned_tensors():
    # Allocate a larger tensor and use a slicing trick to misalign the data pointer.
    batch_size = 4
    in_channels = 3
    out_channels = 8
    kernel_size = 3
    length = 128
    stride = 1
    padding = 1
    dilation = 1
    groups = 1
    bias = False

    # Allocate an input buffer with an extra element to allow misalignment.
    full_x = torch.randn(batch_size, in_channels, length + 1, device="cuda", dtype=torch.float32)
    # Slice off the first element in the last dimension yielding a tensor whose data pointer is offset.
    x = full_x[:, :, 1:]

    # Use PyTorch's Conv1d for reference with the same splitting trick on weight if necessary.
    ref_conv = nn.Conv1d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias
    ).cuda()
    ref_conv.weight.data = torch.randn_like(ref_conv.weight.data).float()
    if bias:
        ref_conv.bias.data = torch.randn_like(ref_conv.bias.data).float()

    # Because x is a sub-tensor, its underlying data pointer may not be 16-byte aligned.
    ref_y = ref_conv(x)

    mod = build_kernel()
    custom_y = mod.forward(
        x,
        ref_conv.weight,
        None if not bias else ref_conv.bias,
        stride,
        padding,
        dilation,
        groups,
    )
    torch.cuda.synchronize()
    # In the misaligned case, results may diverge due to misuse of vectorized loads/stores.
    assert not torch.allclose(custom_y, ref_y, atol=1e-4), "Custom kernel should produce diverging output when tensor addresses are not 16-byte aligned."

# 3. Test for small output lengths (L_out < 4)
def test_small_output():
    # Choose L_in such that the computed L_out is less than 4.
    batch_size = 4
    in_channels = 3
    out_channels = 8
    kernel_size = 3
    # With padding=0, dilation=1, stride=1, to have L_out < 4, let L_in be 4.
    length = 4
    stride = 1
    padding = 0
    dilation = 1
    groups = 1
    bias = False

    ref_conv = nn.Conv1d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias
    ).cuda()
    ref_conv.weight.data = torch.randn_like(ref_conv.weight.data).float()
    if bias:
        ref_conv.bias.data = torch.randn_like(ref_conv.bias.data).float()
    x = torch.randn(batch_size, in_channels, length, device="cuda", dtype=torch.float32)
    ref_y = ref_conv(x)

    mod = build_kernel()
    custom_y = mod.forward(
        x,
        ref_conv.weight,
        None if not bias else ref_conv.bias,
        stride,
        padding,
        dilation,
        groups,
    )
    torch.cuda.synchronize()
    # Even though the kernel has boundary checks, the vectorization of 4 outputs might not correctly handle very small L_out.
    assert not torch.allclose(custom_y, ref_y, atol=1e-4), "Custom kernel is expected to fail with small output lengths (L_out < 4) due to vectorized processing issues."

# 4. Test for the inconsistency introduced by having an unused vectorized load function.
# While this issue does not necessarily trigger a runtime error, we include a test that simply calls the forward function to make sure
# the intention to vectorize both load and store is not inadvertently hidden by only testing store.
def test_vectorized_load_unused():
    # This test checks that the forward function runs without error even though the vectorized load helper is defined but not used.
    batch_size = 4
    in_channels = 3
    out_channels = 8
    kernel_size = 3
    length = 64
    stride = 1
    padding = 1
    dilation = 1
    groups = 1
    bias = True

    ref_conv = nn.Conv1d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias
    ).cuda()
    ref_conv.weight.data = torch.randn_like(ref_conv.weight.data).float()
    if bias:
        ref_conv.bias.data = torch.randn_like(ref_conv.bias.data).float()
    x = torch.randn(batch_size, in_channels, length, device="cuda", dtype=torch.float32)
    
    mod = build_kernel()
    # Simply calling forward to confirm that the kernel linking is not silently broken by the unused load function.
    custom_y = mod.forward(
        x,
        ref_conv.weight,
        ref_conv.bias if bias else None,
        stride,
        padding,
        dilation,
        groups,
    )
    torch.cuda.synchronize()
    # We do not compare output values here because the issue is conceptual (unused code) but report if no runtime error occurs.
    assert custom_y is not None, "Kernel forward call failed when using the unused vectorized load code."

if __name__ == "__main__":
    pytest.main([__file__])
