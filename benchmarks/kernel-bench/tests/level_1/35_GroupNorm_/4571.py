
import torch
import pytest
from torch.utils.cpp_extension import load

# Build the CUDA kernel extension from kernel.cu.
def build_kernel():
    return load(
        name="groupnorm_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )

# A helper function that implements group normalization with PyTorch for reference.
def pytorch_group_norm(x, weight, bias, num_groups, eps):
    # Use torch.nn.functional.group_norm to compute the reference.
    return torch.nn.functional.group_norm(x, num_groups, weight=weight, bias=bias, eps=eps)

# Test case 1: Channels not divisible by num_groups.
# This should cause the kernel to compute incorrect grouping.
def test_channel_divisibility_issue():
    torch.cuda.manual_seed(0)
    # Setup: channels = 10, num_groups = 3 (10 % 3 != 0).
    batch_size = 4
    channels = 10
    height, width = 16, 16
    num_groups = 3
    eps = 1e-5

    x = torch.randn(batch_size, channels, height, width, device='cuda', dtype=torch.float32).contiguous()
    weight = torch.randn(channels, device='cuda', dtype=torch.float32)
    bias = torch.randn(channels, device='cuda', dtype=torch.float32)

    my_module = build_kernel()
    # Running the custom CUDA kernel.
    y_kernel = my_module.forward(x, weight, bias, num_groups, eps)
    # Using PyTorch built-in group norm as reference.
    y_ref = pytorch_group_norm(x, weight, bias, num_groups, eps)

    # The outputs should differ due to misgrouping. We trigger the issue if they are close.
    assert not torch.allclose(y_kernel, y_ref, atol=1e-5), (
        "Test failed: Kernel output matches reference even when channels are not divisible by num_groups, "
        "but it should produce an error/mismatch."
    )

# Test case 2: Non-contiguous input tensor.
def test_non_contiguous_input_issue():
    torch.cuda.manual_seed(0)
    batch_size = 4
    channels = 8  # 8 is divisible by num_groups (here, choose num_groups=4)
    height, width = 16, 16
    num_groups = 4
    eps = 1e-5

    # Create a contiguous tensor and then make it non-contiguous by transposing.
    x_orig = torch.randn(batch_size, channels, height, width, device='cuda', dtype=torch.float32)
    # Transpose such that the tensor is non-contiguous in memory.
    x = x_orig.transpose(1, 2)  # shape becomes (batch_size, height, channels, width)
    # We then force contiguous channels if necessary by reshaping back but with original
    # metadata: here, we simulate a case where our kernel will use x.data_ptr as if it were contiguous.
    # For this test, we do not call contiguous() to trigger the error.
    weight = torch.randn(channels, device='cuda', dtype=torch.float32)
    bias = torch.randn(channels, device='cuda', dtype=torch.float32)

    my_module = build_kernel()
    y_kernel = my_module.forward(x, weight, bias, num_groups, eps)
    # For reference, we call PyTorch group norm which supports non-contiguous inputs.
    y_ref = pytorch_group_norm(x, weight, bias, num_groups, eps)

    # Because the kernel does not handle non-contiguous memory, we expect a discrepancy.
    assert not torch.allclose(y_kernel, y_ref, atol=1e-5), (
        "Test failed: Kernel output is close to reference even with non-contiguous input, "
        "but the kernel assumes contiguous memory and should produce an error/mismatch."
    )

# Test case 3: Unsupported float16 (half precision) input.
def test_half_precision_issue():
    torch.cuda.manual_seed(0)
    batch_size = 4
    channels = 8
    height, width = 16, 16
    num_groups = 4
    eps = 1e-5

    x = torch.randn(batch_size, channels, height, width, device='cuda', dtype=torch.float16).contiguous()
    weight = torch.randn(channels, device='cuda', dtype=torch.float16)
    bias = torch.randn(channels, device='cuda', dtype=torch.float16)

    my_module = build_kernel()
    # The kernel dispatch uses AT_DISPATCH_FLOATING_TYPES which does not include half.
    # We expect the kernel to raise an error when a half-precision tensor is passed.
    with pytest.raises(RuntimeError):
        _ = my_module.forward(x, weight, bias, num_groups, eps)
