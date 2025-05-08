
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build and load the CUDA extension
def build_kernel():
    # Ensure full path is provided so that compilation works correctly in testing environments.
    src_file = os.path.join(os.path.dirname(__file__), "kernel.cu")
    cuda_module = load(
        name="custom_conv_transpose3d",
        sources=[src_file],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: The kernel is not a custom implementation.
# Test: Check if the forward call produces the same result as at::conv_transpose3d (i.e. no custom behavior).
def test_custom_kernel_not_applied():
    module = build_kernel()
    device = "cuda"
    x = torch.randn(2, 4, 8, 8, 8, device=device, dtype=torch.float32)
    # Create a weight tensor with appropriate dimensions
    # For ConvTranspose3d, weight shape is (in_channels, out_channels/groups, kD, kH, kW)
    weight = torch.randn(4, 2, 3, 3, 3, device=device, dtype=torch.float32)
    # Use no bias
    stride = [1, 1, 1]
    padding = [1, 1, 1]
    output_padding = [0, 0, 0]
    groups = 1
    # Call our kernel's forward
    result = module.forward(x, weight, None, stride, padding, output_padding, groups)
    # Call torch's native conv_transpose3d for reference
    ref = torch.nn.functional.conv_transpose3d(x, weight, bias=None,
                                               stride=stride, padding=padding,
                                               output_padding=output_padding, groups=groups)
    # The results should match exactly since the "custom" kernel is a wrapper.
    assert torch.allclose(result, ref), "Custom kernel wrapper does not match aten::conv_transpose3d result."

# Issue 2: The kernel does not validate input tensor dimensions.
def test_input_dimension_validation():
    module = build_kernel()
    device = "cuda"
    # Create a tensor with wrong dimensions (4D instead of 5D)
    x = torch.randn(2, 4, 8, 8, device=device, dtype=torch.float32)
    weight = torch.randn(4, 2, 3, 3, 3, device=device, dtype=torch.float32)
    stride = [1, 1, 1]
    padding = [1, 1, 1]
    output_padding = [0, 0, 0]
    groups = 1

    with pytest.raises(RuntimeError):
        # Expect an error due to wrong tensor dimensions.
        module.forward(x, weight, None, stride, padding, output_padding, groups)

# Issue 3: The kernel does not verify that stride, padding, and output_padding vectors have exactly three elements.
@pytest.mark.parametrize("param_name,param_val", [
    ("stride", [1, 1]),  # too few
    ("padding", [1, 1, 1, 1]),  # too many
    ("output_padding", [0])  # too few
])
def test_parameter_vector_length(param_name, param_val):
    module = build_kernel()
    device = "cuda"
    x = torch.randn(2, 4, 8, 8, 8, device=device, dtype=torch.float32)
    weight = torch.randn(4, 2, 3, 3, 3, device=device, dtype=torch.float32)
    # Set default values for parameters
    stride = [1, 1, 1]
    padding = [1, 1, 1]
    output_padding = [0, 0, 0]
    groups = 1

    # Override the parameter under test with an incorrect vector
    if param_name == "stride":
        stride = param_val
    elif param_name == "padding":
        padding = param_val
    elif param_name == "output_padding":
        output_padding = param_val

    with pytest.raises(Exception):
        module.forward(x, weight, None, stride, padding, output_padding, groups)

# Issue 4: The kernel strictly enforces that inputs be contiguous.
def test_non_contiguous_input():
    module = build_kernel()
    device = "cuda"
    # Create a contiguous tensor first then make it noncontiguous by transposing some dimensions.
    x = torch.randn(2, 4, 8, 8, 8, device=device, dtype=torch.float32)
    x_non_contiguous = x.transpose(1, 2)  # now non contiguous
    weight = torch.randn(4, 2, 3, 3, 3, device=device, dtype=torch.float32)
    stride = [1, 1, 1]
    padding = [1, 1, 1]
    output_padding = [0, 0, 0]
    groups = 1

    with pytest.raises(RuntimeError):
        module.forward(x_non_contiguous, weight, None, stride, padding, output_padding, groups)

# Issue 5: Handling of non-contiguous bias.
def test_non_contiguous_bias():
    module = build_kernel()
    device = "cuda"
    x = torch.randn(2, 4, 8, 8, 8, device=device, dtype=torch.float32)
    weight = torch.randn(4, 2, 3, 3, 3, device=device, dtype=torch.float32)
    bias = torch.randn(4, device=device, dtype=torch.float32)
    # Make bias non-contiguous artificially
    bias = bias.unsqueeze(0).transpose(0, 0).squeeze(0)
    # Note: even though bias is originally contiguous, the test forces the CHECK_INPUT to be triggered by constructing a tensor believed to be non-contiguous.
    # In practice, to simulate non-contiguity, we slice the tensor.
    bias_noncontig = bias[::1]  # This might still be contiguous; let's force non-contiguity:
    bias_noncontig = bias_noncontig.as_strided(bias_noncontig.size(), [100])
    
    stride = [1, 1, 1]
    padding = [1, 1, 1]
    output_padding = [0, 0, 0]
    groups = 1

    with pytest.raises(RuntimeError):
        module.forward(x, weight, bias_noncontig, stride, padding, output_padding, groups)
