
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the CUDA extension from kernel.cu
    kernel_module = load(
        name="kern_ext",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return kernel_module

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_stride_limitation():
    # Issue 1: The kernel accepts only a single stride value.
    # Prepare a scenario with asymmetric convolution strides.
    kernel_module = build_kernel()
    # Create a simple conv input
    x = torch.randn(1, 1, 10, 10, device='cuda')
    weight = torch.randn(1, 1, 3, 3, device='cuda')
    bias = torch.randn(1, device='cuda')
    # Native conv2d supports asymmetric stride (tuple), e.g. (1,2)
    native_out = F.conv2d(x, weight, bias, stride=(1, 2), padding=(1, 1), dilation=(1, 1))
    # The custom kernel only accepts an int for stride so we can only pass 1.
    custom_out = kernel_module.forward(x, weight, bias, 1, (1, 1), (1, 1))
    # Because asymmetric stride is not supported, the output shapes should differ.
    assert native_out.shape != custom_out.shape, "Expected output shapes to differ due to fixed stride limitation."

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_dtype_mismatch():
    # Issue 2: The kernel processes data as float32 only.
    # If we pass a tensor of type float64, the data pointer cast inside the kernel
    # will be incorrect and lead to wrong results or runtime error.
    kernel_module = build_kernel()
    # Construct double precision tensors.
    x = torch.randn(16, 3, 32, 32, device='cuda', dtype=torch.float64)
    weight = torch.randn(64, 3, 3, 3, device='cuda', dtype=torch.float64)
    bias = torch.randn(64, device='cuda', dtype=torch.float64)
    # The forward function does not check for data type, but the kernel expects float32.
    # We expect this misuse to lead to a runtime error (or at least incorrect behavior).
    with pytest.raises(RuntimeError):
        kernel_module.forward(x, weight, bias, 1, (1, 1), (1, 1))

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_bias_shape_mismatch():
    # Issue 3: No bias shape check is performed.
    # Provide a bias tensor whose shape does not match the expected out_channels.
    kernel_module = build_kernel()
    x = torch.randn(1, 3, 10, 10, device='cuda')
    # Correct weight shape would require out_channels = 4; we deliberately set bias shape incorrectly.
    weight = torch.randn(4, 3, 3, 3, device='cuda')
    wrong_bias = torch.randn(2, device='cuda')  # Incorrect shape
    # Expect that using a bias tensor with a mismatched shape causes an error.
    with pytest.raises(RuntimeError):
        kernel_module.forward(x, weight, wrong_bias, 1, (1, 1), (1, 1))

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_unroll_pragma_effect():
    # Issue 4: The unconditional use of "#pragma unroll" may not work optimally when the kernel width
    # is not a compile-time constant. This test uses an unconventional kernel size.
    kernel_module = build_kernel()
    x = torch.randn(1, 1, 15, 15, device='cuda')
    # Use a kernel with a non-standard (and not necessarily compile-time constant) width.
    weight = torch.randn(1, 1, 3, 4, device='cuda')
    bias = torch.randn(1, device='cuda')
    custom_out = kernel_module.forward(x, weight, bias, 1, (1, 2), (1, 1))
    # Compute native conv2d result for reference.
    native_out = F.conv2d(x, weight, bias, stride=1, padding=(1, 2), dilation=(1, 1))
    # In the presence of a problematic unroll policy, the outputs are expected not to match.
    assert not torch.allclose(custom_out, native_out, atol=1e-4), "Kernel output unexpectedly matches native output, suggesting unroll issues."
