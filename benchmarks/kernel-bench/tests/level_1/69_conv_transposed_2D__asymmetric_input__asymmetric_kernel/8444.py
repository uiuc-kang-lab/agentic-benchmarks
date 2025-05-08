
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="custom_conv_transpose2d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test case 1: Groups issue
def test_grouped_convolution():
    # Create a scenario with groups > 1
    batch_size = 2
    in_channels = 4
    out_channels = 4
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = (1, 1)
    output_padding = (0, 0)
    dilation = (1, 1)
    groups = 2  # grouped convolution
    
    # Build input
    H_in, W_in = 8, 8
    x = torch.randn(batch_size, in_channels, H_in, W_in, device="cuda", dtype=torch.float32)
    
    # Create weights and bias for the custom kernel.
    # PyTorch's ConvTranspose2d with groups arranges weight in shape (in_channels, out_channels//groups, kH, kW)
    weight = torch.randn(in_channels, out_channels // groups, kernel_size[0], kernel_size[1], device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)
    
    # Compute expected output using PyTorch's native module
    ref_conv = torch.nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
        bias=True
    ).to(x.device)
    # Inject our custom weight and bias into the PyTorch module.
    with torch.no_grad():
        ref_conv.weight.copy_(weight)
        ref_conv.bias.copy_(bias)
    expected = ref_conv(x)

    # Now call our custom CUDA kernel via the compiled extension
    cuda_module = build_kernel()
    # IMPORTANT: our kernel expects weight to be arranged as if groups==1, i.e. shape (in_channels, out_channels, kH, kW)
    # To simulate the mismatch, we unsqueeze the group dimension (which is not what our kernel supports)
    # We combine the groups by simply repeating the weight tensor incorrectly.
    wrong_weight = weight.view(groups, in_channels // groups, out_channels // groups, kernel_size[0], kernel_size[1])
    # Rearrange into a shape expected by the kernel
    wrong_weight = wrong_weight.reshape(in_channels, out_channels, kernel_size[0], kernel_size[1]).contiguous()
    
    output = cuda_module.forward(x, wrong_weight, bias, list(stride), list(padding), list(output_padding), list(dilation), groups)
    torch.cuda.synchronize()
    
    # Since the kernel ignores groups, the output is expected to differ significantly from the reference.
    assert not torch.allclose(output, expected, atol=1e-5), \
        "Test failed: The custom kernel unexpectedly produced correct output for grouped convolution."

# Test case 2: Data type assumption (float32 only)
def test_non_float_input():
    batch_size = 2
    in_channels = 3
    height = 8
    width = 8
    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float64)  # double tensor
    
    # Dummy parameters for minimal valid kernel invocation.
    weight = torch.randn(in_channels, 3, 3, 3, device="cuda", dtype=torch.float64)
    bias = torch.randn(3, device="cuda", dtype=torch.float64)
    
    stride = [1, 1]
    padding = [1, 1]
    output_padding = [0, 0]
    dilation = [1, 1]
    groups = 1

    cuda_module = build_kernel()
    
    with pytest.raises(RuntimeError):
        # The kernel explicitly calls data_ptr<float>(), so passing a double tensor should trigger a type error.
        output = cuda_module.forward(x, weight, bias, stride, padding, output_padding, dilation, groups)
        torch.cuda.synchronize()

# Test case 3: Non-contiguous input
def test_non_contiguous_input():
    batch_size = 2
    in_channels = 3
    height = 8
    width = 8
    # Create a contiguous tensor and then make it non-contiguous by transposing spatial dimensions.
    x = torch.randn(batch_size, in_channels, height, width, device="cuda", dtype=torch.float32).transpose(2, 3)
    # Force non-contiguity check
    assert not x.is_contiguous(), "Input tensor is unexpectedly contiguous."

    weight = torch.randn(in_channels, 3, 3, 3, device="cuda", dtype=torch.float32)
    bias = torch.randn(3, device="cuda", dtype=torch.float32)
    
    stride = [1, 1]
    padding = [1, 1]
    output_padding = [0, 0]
    dilation = [1, 1]
    groups = 1
    
    cuda_module = build_kernel()
    # The kernel uses data_ptr<float>() which assumes contiguous memory.
    # The output will be computed from the base pointer of a non-contiguous tensor which is likely to be incorrect.
    output = cuda_module.forward(x, weight, bias, stride, padding, output_padding, dilation, groups)
    torch.cuda.synchronize()
    
    # Compare with PyTorch's native ConvTranspose2d (which handles non-contiguity properly)
    conv = torch.nn.ConvTranspose2d(
        in_channels,
        3,
        kernel_size=(3, 3),
        stride=tuple(stride),
        padding=tuple(padding),
        output_padding=tuple(output_padding),
        dilation=tuple(dilation),
        groups=groups,
        bias=True
    ).to(x.device)
    with torch.no_grad():
        conv.weight.copy_(weight)
        conv.bias.copy_(bias)
    expected = conv(x)
    
    # We expect a mismatch due to the kernel assuming contiguity.
    assert not torch.allclose(output, expected, atol=1e-5), \
        "Test failed: The custom kernel unexpectedly produced correct output for non-contiguous input."

if __name__ == "__main__":
    pytest.main([__file__])
