
import torch
import pytest
from torch.nn import functional as F
from torch.utils.cpp_extension import load

# Utility function to compile and load the CUDA extension.
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Helper convolution parameters for tests.
def get_params(batch_size=2, in_channels=4, out_channels=300, kernel_size=3, stride=1, padding=0, output_padding=0, groups=1):
    # Note: out_channels must match the expected bias size for conv_transpose2d.
    # When groups != 1, PyTorch conv_transpose2d weight has shape [in_channels, out_channels//groups, k, k],
    # and the bias (if not None) must be of length out_channels.
    weight = torch.randn(in_channels, out_channels // groups, kernel_size, kernel_size, device='cuda', dtype=torch.float32)
    bias = torch.randn(out_channels, device='cuda', dtype=torch.float32)
    x = torch.randn(batch_size, in_channels, 16, 16, device='cuda', dtype=torch.float32)
    return x, weight, bias, stride, padding, output_padding, groups

# Issue 1: Shared Memory Loading Limitation
# Test case: if out_channels (C_out) > blockDim.x used in the kernel launch,
# then not all bias elements will be loaded and the output will be biased incorrectly.
def test_shared_memory_loading_limitation():
    # Use out_channels=300 (which is >256, the hard-coded block size) to force the problem.
    x, weight, bias, stride, padding, output_padding, groups = get_params(out_channels=300)
    # Compute reference using torch.nn.functional.conv_transpose2d.
    # Note: to emulate our extension without the extra bias, remove bias from the conv call.
    ref = F.conv_transpose2d(x, weight, bias, stride=stride, padding=padding,
                             output_padding=output_padding, groups=groups)
    # Build our extension which calls conv_transpose2d then applies our CUDA kernel.
    mod = build_kernel()
    ext_out = mod.forward(x, weight, bias, stride, padding, output_padding, groups)
    # Because at::conv_transpose2d has already applied bias and our kernel added bias a second time,
    # the error is compounded by the faulty shared memory load.
    # Here we only check that the result is incorrect.
    # We expect a significant discrepancy because some bias values were not loaded.
    diff = (ext_out - ref).abs().mean().item()
    assert diff > 1e-3, f"Test failed: shared memory loading issue not detected, diff = {diff}"

# Issue 2: Incorrect Handling of Groups
# Test case: use groups > 1 so that the actual number of output channels is not equal to weight.size(1).
# The incorrect bias indexing in the kernel should lead to a discrepancy relative to the reference.
def test_incorrect_handling_of_groups():
    groups = 2
    out_channels = 8  # total output channels (should be weight.size(1)*groups).
    in_channels = 4
    # For groups > 1, weight shape: [in_channels, out_channels//groups, k, k]
    kernel_size = 3
    batch_size = 2
    height = 16
    width = 16
    weight = torch.randn(in_channels, out_channels // groups, kernel_size, kernel_size, device='cuda', dtype=torch.float32)
    bias = torch.randn(out_channels, device='cuda', dtype=torch.float32)
    x = torch.randn(batch_size, in_channels, height, width, device='cuda', dtype=torch.float32)
    
    ref = F.conv_transpose2d(x, weight, bias, stride=1, padding=0, output_padding=0, groups=groups)
    mod = build_kernel()
    ext_out = mod.forward(x, weight, bias, 1, 0, 0, groups)
    # The bias is likely added in the wrong channels, so there should be differences.
    diff = (ext_out - ref).abs().mean().item()
    assert diff > 1e-3, f"Test failed: groups handling issue not detected, diff = {diff}"

# Issue 3: Data Type Incompatibility
# Test: Supply a bias tensor in double precision. The kernel expects float and will misinterpret the memory.
def test_data_type_incompatibility():
    x, weight, bias, stride, padding, output_padding, groups = get_params()
    # Make bias double
    bias = bias.double()
    mod = build_kernel()
    with pytest.raises(RuntimeError):
        # Expect a runtime error because the kernel uses data_ptr<float>() on a double tensor.
        _ = mod.forward(x, weight, bias, stride, padding, output_padding, groups)

# Issue 4: Double Bias Addition
# Test: The extension calls at::conv_transpose2d with bias and then adds bias again via the kernel.
# As a result, the output will have the bias added twice compared to the standard implementation.
def test_double_bias_addition():
    x, weight, bias, stride, padding, output_padding, groups = get_params(out_channels=32)
    # Compute the reference using conv_transpose2d with bias applied only once.
    ref = F.conv_transpose2d(x, weight, bias, stride=stride, padding=padding,
                             output_padding=output_padding, groups=groups)
    mod = build_kernel()
    ext_out = mod.forward(x, weight, bias, stride, padding, output_padding, groups)
    # The expected error is exactly one additional bias addition per channel.
    # We manually compute the extra bias across spatial dimensions:
    N, C, H, W = ext_out.shape
    # Create a bias tensor broadcast to output shape and compare
    bias_broadcast = bias.view(1, C, 1, 1).expand(N, C, H, W)
    # Since conv_transpose2d already added the bias, our kernel adds bias again:
    expected = ref + bias_broadcast
    diff = (ext_out - expected).abs().max().item()
    assert diff < 1e-4, f"Test failed: expected double bias addition, but got diff = {diff}"
