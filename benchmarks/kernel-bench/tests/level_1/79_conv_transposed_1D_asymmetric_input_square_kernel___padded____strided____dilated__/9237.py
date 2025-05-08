
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

# Utility: Built-in reference implementation using PyTorch's ConvTranspose1d.
class RefConvTranspose1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
        super(RefConvTranspose1d, self).__init__()
        self.conv_transpose = torch.nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias
        )
        # Copy the weights and bias if provided to later compare to our kernel result.
        # Note: Our kernel expects weight shape (in_channels, out_channels, kernel_size)
        # whereas PyTorch stores it as (in_channels, out_channels, kernel_size), so they match.
    
    def forward(self, x):
        return self.conv_transpose(x)

# Issue 1: Kernel assumes float32 data type. Passing a tensor with a different dtype (e.g., float64)
# will lead to incorrect behavior. We trigger this by using double tensors.
def test_dtype_mismatch():
    cuda_mod = build_kernel()
    batch_size = 4
    in_channels = 3
    out_channels = 2
    kernel_size = 3
    length = 10
    stride = 1
    padding = 1
    dilation = 1
    bias_flag = True

    # Create inputs with double precision.
    x = torch.randn(batch_size, in_channels, length, dtype=torch.float64, device="cuda")
    # Create weight with shape (in_channels, out_channels, kernel_size) in double precision.
    weight = torch.randn(in_channels, out_channels, kernel_size, dtype=torch.float64, device="cuda")
    # Create bias if needed.
    bias = torch.randn(out_channels, dtype=torch.float64, device="cuda") if bias_flag else None

    # Expect failure or incorrect result because the kernel uses data_ptr<float>()
    with pytest.raises(RuntimeError):
        # This should raise a runtime error due to dtype mismatch or lead to a failure during kernel execution.
        out = cuda_mod.forward(x, weight, bias, stride, padding, dilation)
        torch.cuda.synchronize()


# Issue 2: The use of "#pragma unroll" on loops with runtime-determined bounds (kernel_size, in_channels)
# may cause problems for non-typical values. We trigger this issue by using a relatively large kernel_size.
def test_large_kernel_size():
    cuda_mod = build_kernel()
    batch_size = 2
    in_channels = 4
    out_channels = 3
    kernel_size = 11  # large kernel size to challenge loop unrolling assumptions
    length = 20
    stride = 2
    padding = 2
    dilation = 1
    bias_flag = False

    # Prepare inputs with float32.
    x = torch.randn(batch_size, in_channels, length, dtype=torch.float32, device="cuda")
    weight = torch.randn(in_channels, out_channels, kernel_size, dtype=torch.float32, device="cuda")
    bias = torch.randn(out_channels, dtype=torch.float32, device="cuda") if bias_flag else None

    # Run our custom kernel.
    out_kernel = cuda_mod.forward(x, weight, bias, stride, padding, dilation)
    torch.cuda.synchronize()

    # Reference: using PyTorch's built-in ConvTranspose1d.
    ref_conv = RefConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias_flag)
    # Copy the same weights (and bias if exists) for fair comparison.
    with torch.no_grad():
        ref_conv.conv_transpose.weight.copy_(weight)
        if bias_flag:
            ref_conv.conv_transpose.bias.copy_(bias)
    out_ref = ref_conv(x)

    # Since numerical differences might occur, check if results are close.
    assert torch.allclose(out_kernel, out_ref, atol=1e-4), f"Output mismatch for large kernel size. Max diff: {(out_kernel - out_ref).abs().max().item()}"


# Issue 3: Kernel does not validate that stride is nonzero.
def test_stride_zero():
    cuda_mod = build_kernel()
    batch_size = 2
    in_channels = 3
    out_channels = 2
    kernel_size = 3
    length = 10
    stride = 0  # invalid stride, should trigger division-by-zero
    padding = 1
    dilation = 1
    bias_flag = False

    x = torch.randn(batch_size, in_channels, length, dtype=torch.float32, device="cuda")
    weight = torch.randn(in_channels, out_channels, kernel_size, dtype=torch.float32, device="cuda")
    bias = None

    with pytest.raises(RuntimeError):
        out = cuda_mod.forward(x, weight, bias, stride, padding, dilation)
        torch.cuda.synchronize()
