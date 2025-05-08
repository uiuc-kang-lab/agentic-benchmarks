
import torch
import pytest
from torch.utils.cpp_extension import load

# Build the CUDA extension from kernel.cu.
def build_kernel():
    return load(
        name="depthwise_conv2d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

# Helper: canonical depthwise convolution using PyTorch nn.Conv2d
def reference_depthwise_conv2d(x, weight, bias, stride, padding, groups):
    # This only works when groups equals in_channels.
    # For groups ≠ in_channels, we simulate the effect by splitting channels.
    if groups != x.size(1):
        # For simplicity, raise an error to denote unsupported case.
        raise ValueError("Reference implementation supports only groups==in_channels")
    conv = torch.nn.Conv2d(
        in_channels=x.size(1),
        out_channels=x.size(1),
        kernel_size=weight.shape[-1],
        stride=stride,
        padding=padding,
        groups=x.size(1),
        bias=bias is not None,
    ).to(x.device).type_as(x)
    # Manually set weights and bias for a fair comparison.
    conv.weight.data.copy_(weight.view_as(conv.weight.data))
    if bias is not None:
        conv.bias.data.copy_(bias)
    return conv(x)

@pytest.fixture(scope="module")
def cuda_extension():
    return build_kernel()

# Test 1: Trigger issue with groups parameter not being used.
def test_groups_mismatch(cuda_extension):
    # Create an input tensor for which groups != in_channels.
    batch_size, in_channels, height, width = 4, 8, 16, 16
    stride, padding = 1, 0
    kernel_size = 3

    # Use a weight shaped for depthwise convolution: (in_channels, 1, kernel_size, kernel_size)
    weight = torch.randn(in_channels, 1, kernel_size, kernel_size, device='cuda', dtype=torch.float32)
    bias = torch.randn(in_channels, device='cuda', dtype=torch.float32)
    # Create an input tensor.
    x = torch.randn(batch_size, in_channels, height, width, device='cuda', dtype=torch.float32)

    # Here, we purposely set groups to a value different from in_channels.
    groups = 2
    # Run the CUDA kernel (which ignores groups parameter)
    out = cuda_extension.forward(x, weight, bias, stride, padding, groups)
    # Reference implementation can only be computed if groups == in_channels,
    # thus we expect that when groups != in_channels, the CUDA kernel output deviates.
    with pytest.raises(ValueError):
        out_ref = reference_depthwise_conv2d(x, weight, bias, stride, padding, groups)
    # Additionally, we can check that the result of the CUDA kernel does not match
    # what would be expected from a correct depthwise conv (if groups were used correctly).
    out_correct = reference_depthwise_conv2d(x, weight, bias, stride, padding, in_channels)
    assert not torch.allclose(out, out_correct, atol=1e-5), (
        "Kernel output unexpectedly matches a proper depthwise conv result for mismatched groups."
    )

# Test 2: Trigger issue with half precision support.
def test_half_precision_not_supported(cuda_extension):
    batch_size, in_channels, height, width = 4, 3, 16, 16
    stride, padding = 1, 0
    kernel_size = 3

    x = torch.randn(batch_size, in_channels, height, width, device='cuda', dtype=torch.half)
    weight = torch.randn(in_channels, 1, kernel_size, kernel_size, device='cuda', dtype=torch.half)
    bias = torch.randn(in_channels, device='cuda', dtype=torch.half)
    
    with pytest.raises(RuntimeError):
        # Expect a RuntimeError or some dispatch failure because half precision
        # is not covered by the current AT_DISPATCH_FLOATING_TYPES.
        cuda_extension.forward(x, weight, bias, stride, padding, groups=in_channels)

# Test 3: Trigger issue with potential misalignment of tensors.
def test_misaligned_memory(cuda_extension):
    batch_size, in_channels, height, width = 4, 3, 32, 32
    stride, padding = 1, 0
    kernel_size = 3

    # Create a tensor and intentionally slice it to break the typical alignment.
    x_full = torch.randn(batch_size, in_channels, height + 1, width + 1, device='cuda', dtype=torch.float32)
    # Slicing to force a non-contiguous, potentially misaligned tensor.
    x = x_full[:, :, 1:, 1:]
    
    weight = torch.randn(in_channels, 1, kernel_size, kernel_size, device='cuda', dtype=torch.float32)
    bias = torch.randn(in_channels, device='cuda', dtype=torch.float32)
    
    # The misalignment may cause the __ldg() based load to fail or produce performance issues.
    # We compare against a reference implementation.
    out = cuda_extension.forward(x, weight, bias, stride, padding, groups=in_channels)
    out_ref = reference_depthwise_conv2d(x, weight, bias, stride, padding, in_channels)
    # It is possible that misalignment makes the result slightly different or causes errors.
    # Here, if they are not close, we signal an issue.
    assert not torch.allclose(out, out_ref, atol=1e-5), (
        "Kernel output unexpectedly matches the reference despite misaligned input memory; "
        "this indicates the alignment assumption wasn’t enforced."
    )
