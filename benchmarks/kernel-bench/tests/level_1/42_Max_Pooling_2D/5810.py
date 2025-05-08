
import pytest
import torch
from torch.utils.cpp_extension import load

# A helper to build the kernel extension
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1:
# Trigger a compilation/device error by using half precision.
# Because the kernel uses std::numeric_limits<scalar_t>::infinity() and plain max,
# compiling/execution on half precision should fail.
def test_half_precision_not_supported():
    my_module = build_kernel()
    # Create a half precision tensor (float16) even though the kernel expects float/double.
    input_tensor = torch.randn(4, 4, 8, 8, device="cuda", dtype=torch.float16)
    kernel_size = 2
    stride = 2
    padding = 1
    dilation = 1
    with pytest.raises(RuntimeError):
        # This call should error out (or produce a bad result) because half isn't covered by AT_DISPATCH_FLOATING_TYPES.
        my_module.forward(input_tensor, kernel_size, stride, padding, dilation)

# Issue 2:
# Trigger a failure when an integer tensor is passed.
def test_integer_dtype_not_supported():
    my_module = build_kernel()
    # Create an integer tensor which is not a floating type.
    input_tensor = torch.randint(0, 10, (2, 2, 16, 16), device="cuda", dtype=torch.int32)
    kernel_size = 2
    stride = 2
    padding = 0
    dilation = 1
    with pytest.raises(RuntimeError):
        # The kernel dispatch does not include int types.
        my_module.forward(input_tensor, kernel_size, stride, padding, dilation)

# Issue 3:
# No kernel error checking is done. In more complex setups,
# a kernel launch failure (e.g. due to mis‐configured boundaries)
# would go unnoticed. To simulate this, we pass a combination of padding, dilation, and kernel size
# that forces the pooled window boundary computation into a suspicious regime.
def test_pooling_window_boundary_issue():
    my_module = build_kernel()
    # We construct a scenario where dilation is large and padding might cause the computed boundaries to be off.
    batch_size = 1
    channels = 1
    height = 10
    width = 10
    # A combination that is unusual and might trigger boundary mis–computation.
    kernel_size = 3
    stride = 1
    padding = 4  # large padding
    dilation = 3  # large dilation
    input_tensor = torch.randn(batch_size, channels, height, width, device="cuda", dtype=torch.float32)
    
    # Compute output via the custom CUDA kernel.
    output = my_module.forward(input_tensor, kernel_size, stride, padding, dilation)
    # Compute reference output using PyTorch's nn.MaxPool2d with the same settings.
    # Note: This is mainly to illustrate the mismatch that can occur if boundaries are computed incorrectly.
    ref_pool = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride,
                                  padding=padding, dilation=dilation)
    ref_output = ref_pool(input_tensor)
    
    # Check if the outputs are close.
    # In a correct implementation they should agree;
    # if boundaries are computed incorrectly, there will be a noticeable difference.
    assert torch.allclose(output, ref_output, atol=1e-4), (
        f"Output does not match reference output! Max diff: {(output - ref_output).abs().max()}")

# Issue 4:
# Although not a runtime error, using device functions like max and std::numeric_limits
# may result in silent misbehavior. We simulate this by looking at a specific input where the pooling window
# should not be empty. If it is, the kernel returns -infinity.
def test_incorrect_empty_pooling_window():
    my_module = build_kernel()
    # Create an input where the valid pooling window is non-empty.
    batch_size = 1
    channels = 1
    height = 20
    width = 20
    kernel_size = 2
    stride = 2
    # Use a padding/dilation combination that should yield a correct pooling output.
    padding = 1
    dilation = 1
    input_tensor = torch.randn(batch_size, channels, height, width, device="cuda", dtype=torch.float32)
    output = my_module.forward(input_tensor, kernel_size, stride, padding, dilation)
    
    # If due to miscomputed boundaries the pooling window was considered empty,
    # the output would be -infinity.
    # We test that no output element equals negative infinity.
    assert not torch.isinf(output).any(), "Some output elements are -infinity, indicating miscomputed pooling boundaries."
