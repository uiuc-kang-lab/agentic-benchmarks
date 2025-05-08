
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="conv_transpose1d_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.fixture(scope="module")
def kernel_module():
    return build_kernel()

def reference_conv_transpose1d(x, weight, bias, stride, padding, dilation):
    # Use PyTorch's own implementation as reference.
    return torch.nn.functional.conv_transpose1d(x, weight, bias=bias, stride=stride, padding=padding, dilation=dilation)

# Issue 1: Weight constant memory overflow.
# This test uses a weight tensor whose total number of elements exceeds 1024,
# which should trigger the problem.
def test_weight_constant_memory_overflow(kernel_module):
    # Choose dimensions such that: in_channels * out_channels * kernel_width > 1024.
    # For example, in_channels=8, out_channels=16, kernel_width=9 gives 8*16*9 = 1152.
    N = 2
    C_in = 8
    C_out = 16
    L_in = 20
    K_w = 9   # 1152 total elements > 1024.
    stride = 1
    padding = 0
    dilation = 1

    x = torch.randn(N, C_in, L_in, device="cuda", dtype=torch.float32)
    weight = torch.randn(C_in, C_out, K_w, device="cuda", dtype=torch.float32)
    bias = torch.randn(C_out, device="cuda", dtype=torch.float32)

    ref = reference_conv_transpose1d(x, weight, bias, stride, padding, dilation)
    out = kernel_module.forward(x, weight, bias, stride, padding, dilation)
    torch.cuda.synchronize()

    # The expected behavior is that the kernel will produce an incorrect output
    # because the weight doesn't fit in the constant memory array.
    assert not torch.allclose(out, ref, atol=1e-4), (
        "Custom kernel output unexpectedly matches reference despite "
        "weight constant memory overflow issue."
    )

# Issue 2 & 3: Bias usage error when bias is None.
# When bias is not provided, the kernel always reads from constant memory c_bias
# (because its address is never null), which leads to nonzero bias being added.
def test_bias_none_misuse(kernel_module):
    N = 2
    C_in = 3
    C_out = 4
    L_in = 30
    K_w = 5
    stride = 2
    padding = 1
    dilation = 1

    x = torch.randn(N, C_in, L_in, device="cuda", dtype=torch.float32)
    weight = torch.randn(C_in, C_out, K_w, device="cuda", dtype=torch.float32)
    bias = None  # No bias provided

    ref = reference_conv_transpose1d(x, weight, bias, stride, padding, dilation)
    out = kernel_module.forward(x, weight, bias, stride, padding, dilation)
    torch.cuda.synchronize()

    # Because the kernel mistakenly always adds a bias value,
    # the custom output should be different from the reference where no bias is applied.
    assert not torch.allclose(out, ref, atol=1e-4), (
        "Custom kernel output unexpectedly matches reference output when bias is None. "
        "Bias misuse issue not triggered."
    )
