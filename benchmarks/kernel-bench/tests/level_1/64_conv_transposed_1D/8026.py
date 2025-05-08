
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="custom_conv_transpose1d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Test case 1: Trigger constant memory size issue with weights larger than 1024 elements.
def test_constant_memory_limit():
    # Create a weight tensor with more than 1024 elements.
    # For example, for a conv_transpose1d, weight shape is (in_channels, out_channels/groups, kernel_size)
    in_channels = 32
    group_size_out = 8
    kernel_size = 6
    groups = 1
    # Total elements = 32 * 8 * 6 = 1536, which exceeds 1024.
    weight = torch.randn(in_channels, group_size_out, kernel_size, device="cuda", dtype=torch.float32).contiguous()
    # Create an input tensor with a small width.
    batch_size = 4
    input_width = 20
    input_tensor = torch.randn(batch_size, in_channels, input_width, device="cuda", dtype=torch.float32).contiguous()

    # dummy bias tensor as None
    bias = None
    stride = 1
    padding = 0
    output_padding = 0

    module = build_kernel()

    with pytest.raises(RuntimeError) as excinfo:
        # Should trigger the TORCH_CHECK error for weight size exceeding 1024 elements.
        module.forward(input_tensor, weight, bias, stride, padding, output_padding, groups)
    assert "Weight size exceeds constant memory limit" in str(excinfo.value)

# Test case 2: Trigger output_padding issue.
def test_output_padding_mismatch():
    # Use weight with allowed size, e.g. with less than or equal to 1024 elements.
    in_channels = 4
    group_size_out = 3  # out_channels = 3 * groups, groups=1 here
    kernel_size = 3    # total elements = 4*3*3 = 36 (< 1024)
    groups = 1
    weight = torch.randn(in_channels, group_size_out, kernel_size, device="cuda", dtype=torch.float32).contiguous()
    bias = torch.randn(group_size_out * groups, device="cuda", dtype=torch.float32).contiguous()

    batch_size = 2
    input_width = 10
    stride = 2
    padding = 1
    output_padding = 1   # non-zero output_padding
    
    # Build our custom kernel module.
    module = build_kernel()
    # Run our custom kernel.
    out_custom = module.forward(torch.randn(batch_size, in_channels, input_width, device="cuda", dtype=torch.float32).contiguous(),
                                  weight, bias, stride, padding, output_padding, groups)
    
    # Compare with PyTorch's native ConvTranspose1d.
    conv_transpose = torch.nn.ConvTranspose1d(in_channels, group_size_out * groups, kernel_size,
                                                stride=stride, padding=padding, output_padding=output_padding,
                                                bias=True)
    # Set conv_transpose weights and bias to match our test tensors:
    with torch.no_grad():
        conv_transpose.weight.copy_(weight)
        conv_transpose.bias.copy_(bias)
    x = torch.randn(batch_size, in_channels, input_width, device="cuda", dtype=torch.float32)
    out_reference = conv_transpose(x)
    
    # Since the CUDA kernel does not handle output_padding contributions properly,
    # we expect a mismatch between our kernel and the PyTorch reference.
    torch.cuda.synchronize()
    assert not torch.allclose(out_custom, out_reference, atol=1e-5), "Custom kernel output unexpectedly matches PyTorch output despite output_padding mismatch."

# Test case 3: Trigger dtype issue by using non-float32 input.
def test_dtype_issue():
    in_channels = 4
    group_size_out = 2
    kernel_size = 3
    groups = 1
    # Create weight tensor in double precision
    weight = torch.randn(in_channels, group_size_out, kernel_size, device="cuda", dtype=torch.double).contiguous()
    input_width = 15
    batch_size = 3
    # Create input tensor in double precision
    input_tensor = torch.randn(batch_size, in_channels, input_width, device="cuda", dtype=torch.double).contiguous()
    # Create bias in double precision
    bias = torch.randn(group_size_out * groups, device="cuda", dtype=torch.double).contiguous()

    stride = 1
    padding = 0
    output_padding = 0

    module = build_kernel()

    with pytest.raises(RuntimeError):
        # Expect the kernel to fail or produce an error due to unsupported dtype.
        module.forward(input_tensor, weight, bias, stride, padding, output_padding, groups)
