
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="selu_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Non-contiguous input tensor.
def test_non_contiguous_input():
    my_module = build_kernel()
    # Create a contiguous tensor
    x = torch.randn(32, 32, device="cuda", dtype=torch.float32)
    # Make it non-contiguous by transposing
    x_nc = x.t()  # now non-contiguous
    # Compute SELU using the kernel
    out_kernel = my_module.forward(x_nc)
    # Compute SELU using PyTorch's implementation (which handles non-contiguity correctly)
    # .selu() will produce the expected result if the input is interpreted by its strides.
    out_torch = torch.selu(x_nc)
    # Since the kernel assumed contiguous memory, the result will differ.
    # We expect that the results are NOT equal (to expose the issue).
    assert not torch.allclose(out_kernel, out_torch, atol=1e-6), (
        "Kernel output matches expected SELU output for a non-contiguous input, but it should be incorrect "
        "because the kernel does not properly handle non-contiguous tensors."
    )

# Issue 2: No kernel launch error checking.
def test_kernel_launch_error_checking():
    my_module = build_kernel()
    # Intentionally provide an input tensor with an incorrect (CPU) device to force failure.
    x_cpu = torch.randn(1024, device="cpu", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="Input tensor must be a CUDA tensor"):
        my_module.forward(x_cpu)
    # Note: It is not trivial to force a kernel launch error from Python,
    # but this test shows that erroneous usage (wrong device) triggers an error.

# Issue 3: Incompatible data type.
def test_incompatible_dtype():
    my_module = build_kernel()
    # Provide a CUDA tensor with wrong data type (float64 instead of float32)
    x_double = torch.randn(1024, device="cuda", dtype=torch.float64)
    with pytest.raises(RuntimeError, match="Input must be float32"):
        my_module.forward(x_double)
