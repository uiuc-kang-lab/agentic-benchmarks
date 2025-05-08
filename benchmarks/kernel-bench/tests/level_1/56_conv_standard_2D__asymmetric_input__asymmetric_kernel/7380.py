
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# Helper function to build the extension using kernel.cu.
def build_kernel():
    cuda_module = load(
        name="custom_conv2d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1: Kernel only supports float32.
# Here we feed half-precision inputs.  The expected behavior is that the kernel
# misinterprets the data (or produces wrong results) because the pointers are cast to float*.
# So we compare our kernel’s output against PyTorch’s F.conv2d in float16 and expect a significant difference.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_non_float32_input():
    cuda_mod = build_kernel()
    # Create half-precision input, weight and bias (if used)
    N, C_in, H, W = 2, 3, 16, 16
    C_out = 8
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = (1, 1)
    dilation = (1, 1)
    groups = 1
    # Use half precision
    x = torch.randn(N, C_in, H, W, device="cuda", dtype=torch.float16)
    # Note: In a proper PyTorch conv2d with groups=1 the weight shape is (C_out, C_in, K_h, K_w)
    weight = torch.randn(C_out, C_in, kernel_size[0], kernel_size[1], device="cuda", dtype=torch.float16)
    # We deliberately pass bias = None
    # Run custom kernel (which wrongly interprets the memory as float32)
    out_custom = cuda_mod.forward(x, weight, None, list(stride), list(padding), list(dilation), groups)
    # Use PyTorch conv2d in half precision as reference:
    out_ref = F.conv2d(x, weight, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups)
    # Because the custom kernel does not dispatch based on dtype, the values will be totally off.
    diff = (out_custom.float() - out_ref.float()).abs().max().item()
    # Expect a large difference.
    assert diff > 1e-3, "Custom kernel unexpectedly produced close results with non-float32 input."

# Issue 2: No error checking after kernel launch.
# We force a kernel launch error by passing a CPU tensor instead of a CUDA tensor.
# (The TORCH_CHECK(input.is_cuda(), ...) in conv2d_cuda should catch this.)
def test_non_cuda_input():
    cuda_mod = build_kernel()
    x = torch.randn(2, 3, 16, 16, device="cpu", dtype=torch.float32)
    weight = torch.randn(8, 3, 3, 3, device="cpu", dtype=torch.float32)
    stride = (1, 1)
    padding = (1, 1)
    dilation = (1, 1)
    groups = 1
    with pytest.raises(RuntimeError):
        cuda_mod.forward(x, weight, None, list(stride), list(padding), list(dilation), groups)

# Issue 3: No verification on the compatibility between weight tensor shape and groups.
# Here we deliberately create a weight tensor that has an incorrect shape.
# For example, if groups > 1 then the weight should have shape [C_out, C_in/groups, K_h, K_w].
# We pass a weight tensor with shape [C_out, C_in, K_h, K_w] (i.e. not divided by groups)
# In such a case we expect the kernel to access out-of-bound memory.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_incorrect_weight_dimensions():
    cuda_mod = build_kernel()
    N, C_in, H, W = 2, 4, 16, 16
    groups = 2  # then C_in/groups = 2; weight should be shape (C_out,2, K_h, K_w)
    C_out = 4
    kernel_size = (3, 3)
    # Create input correctly.
    x = torch.randn(N, C_in, H, W, device="cuda", dtype=torch.float32)
    # Create an incorrectly shaped weight and let the kernel think that the weights are of shape:
    # expected shape: (C_out, C_in/groups, K_h, K_w) but here we give (C_out, C_in, K_h, K_w)
    weight = torch.randn(C_out, C_in, kernel_size[0], kernel_size[1], device="cuda", dtype=torch.float32)
    
    # We also compute a reference using F.conv2d with correct weight shape.
    # Since the shapes differ, we expect the custom kernel to produce nonsensical output possibly causing an
    # out-of-bound access.  (We catch a RuntimeError if the illegal access is detected by CUDA.)
    stride = (1, 1)
    padding = (1, 1)
    dilation = (1, 1)
    with pytest.raises(RuntimeError):
        out = cuda_mod.forward(x, weight, None, list(stride), list(padding), list(dilation), groups)
        # Force synchronization to catch any asynchronous error.
        torch.cuda.synchronize()

# Issue 4: Fixed thread block configuration may not be valid for extremely large output dimensions.
# We trigger an error by creating a tensor with huge spatial dimensions so that N * C_out * H_out * W_out
# leads to a grid size that exceeds what CUDA allows.
# (A RuntimeError is expected because the kernel launch will fail.)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_large_output_config():
    cuda_mod = build_kernel()
    # Use modest batch and channel dimensions but huge spatial dimensions.
    N, C_in = 1, 3
    H, W = 1, 1  # We set small input spatially, but we will use a kernel which outputs extremely large spatial dims by
                # setting padding and dilation accordingly.
    # Let weight be 1x1 so that output dims artificially become huge when using a high dilation.
    C_out = 3
    kernel_size = (1, 1)
    # To force an enormous grid, we simulate output dims by “fooling” the convolution calculation.
    # For instance, set padding such that H_out becomes enormous.
    # (Note: This test may use a lot of resources or fail when run on real hardware.
    # Hence, we expect a RuntimeError.)
    huge = 1 << 29  # extremely huge output width (this number will likely exceed CUDA grid limits)
    # We craft a fake input so that the computed output dims will be huge.
    # Using the output formula: H_out = (H_in + 2*padding - dilation*(K-1)-1)/stride + 1.
    # Choose padding such that H_out is huge.
    padding = (huge, huge)
    dilation = (1, 1)
    stride = (1, 1)
    # Create input and weight.
    x = torch.randn(N, C_in, H, W, device="cuda", dtype=torch.float32)
    weight = torch.randn(C_out, C_in, kernel_size[0], kernel_size[1], device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        out = cuda_mod.forward(x, weight, None, list(stride), list(padding), list(dilation), 1)
        torch.cuda.synchronize()
