
import pytest
import torch
from torch.utils.cpp_extension import load

# Build the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# ------------------------------------------------------------------------------
# Test 1. Data type mismatch: Supply double tensors instead of float tensors.
def test_dtype_mismatch():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cuda_mod = build_kernel()
    B = 2
    in_channels = 3
    in_size = 32
    out_channels = 4
    kernel_size = 3
    stride = 1
    dilation = 1

    # Create input tensors with double type instead of float32.
    x = torch.randn(B, in_channels, in_size, device='cuda', dtype=torch.float64).contiguous()
    weight = torch.randn(out_channels, in_channels, kernel_size, device='cuda', dtype=torch.float64).contiguous()
    bias = torch.randn(out_channels, device='cuda', dtype=torch.float64)

    # Expect the kernel to either raise an error or produce an incorrect result.
    with pytest.raises(RuntimeError):
        cuda_mod.forward(x, weight, bias, stride, dilation)

# ------------------------------------------------------------------------------
# Test 2. Lack of support for padding: Provide a scenario where a non-zero padding is expected.
#
# The kernel does not have a padding parameter so if a user simulates the effect
# of padded input by computing out_size using a padding value, the output will be wrong.
# Here we emulate this by manually padding x and then comparing with the output shape expected
# from a padded convolution.
def test_padding_not_supported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cuda_mod = build_kernel()
    B = 2
    in_channels = 3
    in_size = 32
    out_channels = 4
    kernel_size = 3
    stride = 1
    dilation = 1
    padding = 2  # Non-zero padding which our kernel does not support

    # Create input tensor and pad it manually.
    x = torch.randn(B, in_channels, in_size, device='cuda', dtype=torch.float32).contiguous()
    # Instead of using the kernel to supply a padding value, we simulate convolution with padding
    # by explicitly padding the input.
    x_padded = torch.nn.functional.pad(x, (padding, padding))
    
    weight = torch.randn(out_channels, in_channels, kernel_size, device='cuda', dtype=torch.float32).contiguous()
    bias = torch.randn(out_channels, device='cuda', dtype=torch.float32)

    # Compute expected output size using padded input.
    padded_in_size = in_size + 2*padding
    expected_out_size = (padded_in_size - dilation * (kernel_size - 1) - 1) // stride + 1

    # Use our kernel (which ignores padding) on the original x.
    output = cuda_mod.forward(x, weight, bias, stride, dilation)
    
    # Check that the output size does not match the expected value for a padded convolution.
    assert output.size(2) != expected_out_size, (
        "Kernel incorrectly supports padding even though no padding parameter is accepted."
    )

# ------------------------------------------------------------------------------
# Test 3. Group convolution is not supported.
# We simulate a group convolution scenario by creating a weight tensor that
# has a different second dimension than the number of input channels.
def test_groups_not_supported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cuda_mod = build_kernel()
    B = 2
    in_channels = 4  # suppose actual input channels
    groups = 2
    # For group convolution, each group operates on in_channels/groups channels.
    group_in_channels = in_channels // groups
    out_channels = 6
    kernel_size = 3
    stride = 1
    dilation = 1

    x = torch.randn(B, in_channels, 32, device='cuda', dtype=torch.float32).contiguous()
    # Create weight for group convolution: normally shape is (out_channels, in_channels/groups, kernel_size)
    weight = torch.randn(out_channels, group_in_channels, kernel_size, device='cuda', dtype=torch.float32).contiguous()
    bias = torch.randn(out_channels, device='cuda', dtype=torch.float32)

    # The kernel expects the second dimension of weight to equal in_channels.
    with pytest.raises(RuntimeError):
        cuda_mod.forward(x, weight, bias, stride, dilation)

# ------------------------------------------------------------------------------
# Test 4. Non-contiguous inputs: Supply non-contiguous tensors to trigger the contiguous check.
def test_non_contiguous_input():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cuda_mod = build_kernel()
    B = 2
    in_channels = 3
    in_size = 32
    out_channels = 4
    kernel_size = 3
    stride = 1
    dilation = 1

    x = torch.randn(B, in_channels, in_size, device='cuda', dtype=torch.float32)
    # Make x non-contiguous by a transpose operation and then reshape back.
    x_noncontig = x.transpose(1, 2).contiguous().transpose(1, 2)  # This preserves contiguity,
    # so introduce non-contiguity another way:
    x_noncontig = x[..., ::2].repeat_interleave(2, dim=2)
    if x_noncontig.is_contiguous():
        # Force non-contiguity
        x_noncontig = x_noncontig.as_strided(x_noncontig.size(), [x_noncontig.stride(0), x_noncontig.stride(1), x_noncontig.stride(2) + 1])
    
    weight = torch.randn(out_channels, in_channels, kernel_size, device='cuda', dtype=torch.float32).contiguous()
    bias = torch.randn(out_channels, device='cuda', dtype=torch.float32)

    with pytest.raises(RuntimeError):
        cuda_mod.forward(x_noncontig, weight, bias, stride, dilation)
