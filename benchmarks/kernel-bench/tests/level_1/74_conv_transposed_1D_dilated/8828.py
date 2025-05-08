
import torch
import pytest
from torch.utils.cpp_extension import load
import os

# A helper function to build the CUDA extension.
def build_kernel():
    cuda_module = load(
        name="conv_transpose1d_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test case 1: Trigger issue with weight tensor exceeding the constant memory capacity.
def test_weight_constant_memory_limit():
    # Create a weight tensor that exceeds 4096 elements.
    # For example, use in_channels=4, out_channels=32, kernel_size=50 gives 4*32*50 = 6400 elements.
    in_channels = 4
    out_channels = 32
    kernel_size = 50  # 4 * 32 * 50 = 6400 > 4096
    weight = torch.randn(in_channels, out_channels, kernel_size, device="cuda", dtype=torch.float32)
    x = torch.randn(2, in_channels, 20, device="cuda", dtype=torch.float32)
    # The kernel is expected to raise a TORCH_CHECK error.
    cuda_mod = build_kernel()
    with pytest.raises(RuntimeError, match="Weight size exceeds constant memory capacity"):
        # the forward function takes x, weight, bias, stride, padding, dilation
        cuda_mod.forward(x, weight)

# Test case 2: Trigger issue due to unsupported data type (float64).
def test_input_dtype():
    # Use a weight and input with dtype float64.
    in_channels = 3
    out_channels = 8
    kernel_size = 3
    weight = torch.randn(in_channels, out_channels, kernel_size, device="cuda", dtype=torch.float64)
    x = torch.randn(2, in_channels, 10, device="cuda", dtype=torch.float64)
    cuda_mod = build_kernel()
    with pytest.raises(RuntimeError, match="Input tensor must be on CUDA device"):
        # Actually, the error message is not for dtype but when trying to access data_ptr<float>()
        # the conversion will misinterpret the memory. We expect an error because the kernel assumes float.
        cuda_mod.forward(x, weight)

# Test case 3: Test with parameters that simulate a more complex configuration (e.g. group conv)
# Since the kernel does not support grouping, we try to simulate group behavior
def test_grouped_convolution_behavior():
    # Here we simulate a grouped scenario by creating weights with an additional dimension, which our kernel
    # does not account for. We embed the group dimension in the weight and expect incorrect indexing.
    in_channels = 4  # Suppose actual in_channels per group is 2
    out_channels = 6  # Suppose actual out_channels per group is 3
    kernel_size = 3
    groups = 2
    # Create a weight tensor with shape (groups, in_channels, out_channels, kernel_size)
    weight = torch.randn(groups, in_channels, out_channels, kernel_size, device="cuda", dtype=torch.float32)
    # Flatten the group dimension into the in_channels dimension (this is not equivalent to a real grouped conv)
    weight_flat = weight.view(groups * in_channels, out_channels, kernel_size)
    x = torch.randn(2, groups * in_channels, 10, device="cuda", dtype=torch.float32)
    cuda_mod = build_kernel()
    # Since our kernel does not support groups, the output will be wrong.
    y = cuda_mod.forward(x, weight_flat)
    # We compute a reference using PyTorch (without groups, but with the same flattening) for comparison.
    conv = torch.nn.ConvTranspose1d(
        in_channels=groups * in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        bias=False
    ).to("cuda")
    # Manually assign weight_flat to conv (transposed conv weight ordering may differ).
    conv.weight.data.copy_(weight_flat)
    y_ref = conv(x)
    # The output won't match because our kernel assumes a non-group scenario.
    assert not torch.allclose(y, y_ref, atol=1e-4), "Kernel should not support grouped convolution correctly."

# Test case 4: Check for potential synchronization issue: updating weight asynchronously.
def test_asynchronous_weight_update():
    # In this test, we update the weight tensor on the fly and launch the kernel immediately.
    # While this does not guarantee to always trigger a synchronization issue, it simulates a misuse.
    in_channels = 3
    out_channels = 16
    kernel_size = 3
    weight = torch.randn(in_channels, out_channels, kernel_size, device="cuda", dtype=torch.float32)
    x = torch.randn(2, in_channels, 20, device="cuda", dtype=torch.float32)
    cuda_mod = build_kernel()
    # Launch kernel once to load weight into constant memory.
    y1 = cuda_mod.forward(x, weight)
    # Immediately update weight tensor.
    weight.add_(1.0)
    # Launch kernel again. The constant memory copy was done from an updated weight,
    # but if the cudaMemcpyToSymbol call isn’t synchronized properly, we might experience timing issues.
    y2 = cuda_mod.forward(x, weight)
    # We cannot reliably check for synchronization issues, but we check that the outputs are different.
    assert not torch.allclose(y1, y2, atol=1e-4), "Outputs should differ after weight update if properly synchronized."
