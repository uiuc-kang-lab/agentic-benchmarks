
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# A helper function to create a dummy weight tensor that matches the expected layout (in_channels, out_channels, K, K)
def create_weight(in_channels, out_channels, kernel_size, device, dtype):
    # Weight shape for ConvTranspose2d in PyTorch: (in_channels, out_channels, kernel_size, kernel_size)
    return torch.randn(in_channels, out_channels, kernel_size, kernel_size, device=device, dtype=dtype)

# Issue 1: groups != 1 should trigger an error.
def test_groups_not_supported():
    cuda_module = build_kernel()
    device = "cuda"
    # Create dummy input and weight tensors
    B, C_in, H, W = 2, 4, 16, 16
    C_out = 8
    kernel_size = 3
    stride = 1
    padding = 0

    input_tensor = torch.randn(B, C_in, H, W, device=device, dtype=torch.float32).contiguous()
    weight = create_weight(C_in, C_out, kernel_size, device, torch.float32).contiguous()

    # groups != 1 is not supported so expect a check failure
    with pytest.raises(RuntimeError, match="Only groups==1 is supported"):
        cuda_module.forward(input_tensor, weight, None, stride, padding, 0, 2)

# Issue 2: output_padding != 0 should trigger an error.
def test_output_padding_not_supported():
    cuda_module = build_kernel()
    device = "cuda"
    B, C_in, H, W = 2, 4, 16, 16
    C_out = 8
    kernel_size = 3
    stride = 1
    padding = 0
    output_padding = 1  # non-zero output_padding

    input_tensor = torch.randn(B, C_in, H, W, device=device, dtype=torch.float32).contiguous()
    weight = create_weight(C_in, C_out, kernel_size, device, torch.float32).contiguous()

    with pytest.raises(RuntimeError, match="Only output_padding==0 is supported"):
        cuda_module.forward(input_tensor, weight, None, stride, padding, output_padding, 1)

# Issue 3: Data type other than float32 (e.g., float64) is not supported.
def test_different_dtype():
    cuda_module = build_kernel()
    device = "cuda"
    B, C_in, H, W = 2, 4, 16, 16
    C_out = 8
    kernel_size = 3
    stride = 1
    padding = 0

    # Create tensors of type float64 instead of float32.
    input_tensor = torch.randn(B, C_in, H, W, device=device, dtype=torch.float64).contiguous()
    weight = create_weight(C_in, C_out, kernel_size, device, torch.float64).contiguous()

    # Since the kernel code uses float (float32), the absence of a check means the kernel call will lead to wrong behavior or a runtime error.
    with pytest.raises(RuntimeError):
        cuda_module.forward(input_tensor, weight, None, stride, padding, 0, 1)

# Issue 5: Non-contiguous tensors should trigger an error (as checked explicitly).
def test_non_contiguous_input():
    cuda_module = build_kernel()
    device = "cuda"
    B, C_in, H, W = 2, 4, 16, 16
    C_out = 8
    kernel_size = 3
    stride = 1
    padding = 0

    # Create an input tensor and make it non-contiguous by transposing
    input_tensor = torch.randn(B, C_in, H, W, device=device, dtype=torch.float32).transpose(2, 3)
    weight = create_weight(C_in, C_out, kernel_size, device, torch.float32).contiguous()

    with pytest.raises(RuntimeError, match="Input tensor must be contiguous"):
        cuda_module.forward(input_tensor, weight, None, stride, padding, 0, 1)

# Issue 4: Dilated convolutions are not supported.
# Since the kernel does not accept a dilation parameter, we simulate a case where one might be expected.
# Although the interface itself does not accept dilation, in a more general framework a dilation would be passed.
# Here, we replicate that by mimicking a conv_transpose2d scenario where the effective receptive field would require dilation.
def test_dilated_convolution_not_supported():
    cuda_module = build_kernel()
    device = "cuda"
    B, C_in, H, W = 2, 4, 16, 16
    C_out = 8
    kernel_size = 3
    stride = 1
    padding = 1
    # Suppose we expected a dilation factor of 2, which is not supported by the kernel.
    # We simulate the wrong behavior by comparing against a manual expected output obtained using conv_transpose2d with dilation.
    input_tensor = torch.randn(B, C_in, H, W, device=device, dtype=torch.float32).contiguous()
    weight = torch.randn(C_in, C_out, kernel_size, kernel_size, device=device, dtype=torch.float32).contiguous()
    
    # Use PyTorch's conv_transpose2d with dilation to produce a reference output.
    # Note: PyTorch's conv_transpose2d supports dilation, so the reference and our kernel should differ.
    ref_out = torch.nn.functional.conv_transpose2d(input_tensor, weight, bias=None,
                                                    stride=stride, padding=padding, dilation=2)
    # Our modular kernel does not have dilation and will produce a different output shape.
    out = cuda_module.forward(input_tensor, weight, None, stride, padding, 0, 1)
    
    # The output shape or numerical content will not match due to the unsupported dilation.
    with pytest.raises(AssertionError):
        assert torch.allclose(out, ref_out, atol=1e-5)
