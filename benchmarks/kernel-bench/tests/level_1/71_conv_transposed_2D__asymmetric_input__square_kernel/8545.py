
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

# Issue 1: Input type assumption (only float32 supported)
def test_input_tensor_type():
    # Create tensors in double precision
    batch_size = 2
    in_channels = 4
    out_channels = 6
    kernel_h, kernel_w = 3, 3
    in_h, in_w = 8, 8

    # Create input, weight and bias as float64 tensors
    x = torch.randn(batch_size, in_channels, in_h, in_w, dtype=torch.float64, device="cuda")
    weight = torch.randn(in_channels, out_channels // 1, kernel_h, kernel_w, dtype=torch.float64, device="cuda")
    bias = torch.randn(out_channels, dtype=torch.float64, device="cuda")

    my_module = build_kernel()

    # Expect that using non-float32 tensor may lead to wrong results.
    # Here, we compare to PyTorch's own ConvTranspose2d and expect a mismatch.
    conv_transpose = torch.nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size=(kernel_h, kernel_w),
        stride=1, padding=0, bias=True, groups=1
    ).to(device="cuda", dtype=torch.float64)
    # Manually copy the weights and bias for a fair comparison.
    with torch.no_grad():
        conv_transpose.weight.copy_(weight)
        conv_transpose.bias.copy_(bias)
    
    # Run our CUDA kernel function with float64 inputs (it will reinterpret memory as float32).
    output_cuda = my_module.forward(x, weight, bias, stride=1, padding=0, output_padding=0, groups=1)
    output_pt = conv_transpose(x)
    
    # The outputs should strongly differ due to misinterpreted data representations.
    assert not torch.allclose(output_cuda, output_pt, atol=1e-4), \
        "Kernel incorrectly accepted non-float32 tensor without error!"

# Issue 2: Missing kernel launch error checking.
def test_kernel_launch_error_checking():
    # In this test, we pass deliberately mis-shaped tensors that should trigger an error during kernel execution.
    # For instance, a weight tensor with an invalid shape (e.g., 3 dimensions instead of 4)
    batch_size = 2
    in_channels = 4
    out_channels = 6
    kernel_h, kernel_w = 3, 3
    in_h, in_w = 8, 8

    x = torch.randn(batch_size, in_channels, in_h, in_w, device="cuda", dtype=torch.float32)
    # Creating an invalid weight shape (3D tensor instead of expected 4D tensor)
    weight = torch.randn(in_channels, kernel_h, kernel_w, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    my_module = build_kernel()
    with pytest.raises(IndexError):
        # The kernel should eventually try to access an element out-of-bounds.
        _ = my_module.forward(x, weight, bias, stride=1, padding=0, output_padding=0, groups=1)
        torch.cuda.synchronize()

# Issue 3: Weight tensor dimension validation.
def test_invalid_stride_padding_outputpadding():
    # This test passes an invalid (3-element) sequence for stride.
    batch_size = 2
    in_channels = 4
    out_channels = 6
    kernel_h, kernel_w = 3, 3
    in_h, in_w = 8, 8

    x = torch.randn(batch_size, in_channels, in_h, in_w, device="cuda", dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels // 1, kernel_h, kernel_w, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    my_module = build_kernel()

    # Passing an invalid stride sequence (3 elements instead of 2) should trigger an error in parseIntArrayRef.
    with pytest.raises(RuntimeError, match="Must provide either a single integer or 2 integers"):
        _ = my_module.forward(x, weight, bias, stride=[1, 2, 3], padding=0, output_padding=0, groups=1)
