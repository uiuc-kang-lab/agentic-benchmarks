
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    module = load(
        name="custom_conv2d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return module

# Issue 1: Kernel assumes float32 input.
def test_non_float32_input():
    # Create inputs and weights as float64 (double)
    device = "cuda"
    batch_size, in_channels, height, width = 2, 3, 32, 32
    out_channels = 4
    kernel_h, kernel_w = 3, 3
    
    # Create input and weight tensors of double type.
    x = torch.randn(batch_size, in_channels, height, width, dtype=torch.float64, device=device)
    weight = torch.randn(out_channels, in_channels, kernel_h, kernel_w, dtype=torch.float64, device=device)
    
    # Bias is optional: set to None.
    bias = None
    
    # We force the custom kernel execution by choosing dilation==1 and groups==1.
    mod = build_kernel()
    
    # The custom kernel calls .data_ptr<float>(), so it will reinterpret the double data.
    # Compute output using the custom kernel.
    out_custom = mod.forward(x, weight, bias, 1, 0, 1, 1)
    
    # Compute reference output using torch's conv2d operating in double precision.
    out_ref = torch.nn.functional.conv2d(x, weight, bias, stride=1, padding=0)
    
    # The outputs are expected to differ (or be meaningless) because the kernel misinterprets double data.
    # This test passes if the two outputs are NOT close.
    assert not torch.allclose(out_custom, out_ref, atol=1e-5), (
        "Kernel incorrectly accepted non-float32 input without error."
    )

# Issue 2: Kernel assumes contiguous inputs.
def test_non_contiguous_input():
    device = "cuda"
    batch_size, in_channels, height, width = 2, 3, 32, 32
    out_channels = 4
    kernel_h, kernel_w = 3, 3
    
    # Create a contiguous tensor and then make it non-contiguous via a transpose operation.
    x = torch.randn(batch_size, in_channels, height, width, device=device, dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_h, kernel_w, device=device, dtype=torch.float32)
    x_non_contig = x.transpose(1, 2)  # Now non-contiguous
    x_non_contig = x_non_contig.transpose(1, 2)  # Make shape same but non-contiguous
    
    bias = None
    mod = build_kernel()
    
    # Expect a runtime error because the forward() function checks for contiguity.
    with pytest.raises(RuntimeError, match="Input must be contiguous"):
        mod.forward(x_non_contig, weight, bias, 1, 0, 1, 1)

# Issue 3: Kernel does not support dilation != 1 or groups != 1.
def test_dilation_and_groups_not_supported():
    device = "cuda"
    batch_size, in_channels, height, width = 2, 4, 32, 32
    out_channels = 4
    kernel_h, kernel_w = 3, 3

    # Create inputs and weights in float32 and contiguous.
    x = torch.randn(batch_size, in_channels, height, width, device=device, dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_h, kernel_w, device=device, dtype=torch.float32)
    bias = None

    mod = build_kernel()

    # When dilation != 1 or groups != 1, the custom forward will call torch::conv2d, not the custom kernel.
    # So, the output computed here should equal torch.nn.functional.conv2d with those parameters.
    dilation = 2
    groups = 2
    out_module = mod.forward(x, weight, bias, 1, 0, dilation, groups)
    out_ref = torch.nn.functional.conv2d(x, weight, bias, stride=1, padding=0, dilation=dilation, groups=groups)
    
    assert torch.allclose(out_module, out_ref, atol=1e-5), (
        "Fallback to torch::conv2d for dilation/groups did not produce correct result."
    )

# Issue 4 & 5: These are more subtle. The use of __ldg and ambiguous min/max might lead to
# compilation issues in more complex scenarios. To simulate a stress test on these aspects,
# we run the kernel on larger tensors.
def test_large_tensor_stress():
    device = "cuda"
    batch_size, in_channels, height, width = 8, 3, 128, 128
    out_channels = 16
    kernel_h, kernel_w = 5, 3

    x = torch.randn(batch_size, in_channels, height, width, device=device, dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_h, kernel_w, device=device, dtype=torch.float32)
    bias = torch.randn(out_channels, device=device, dtype=torch.float32)

    mod = build_kernel()
    out_custom = mod.forward(x, weight, bias, 1, 1, 1, 1)
    out_ref = torch.nn.functional.conv2d(x, weight, bias, stride=1, padding=1)

    assert torch.allclose(out_custom, out_ref, atol=1e-4), (
        "Kernel output on large tensor differs significantly from torch::conv2d reference."
    )
