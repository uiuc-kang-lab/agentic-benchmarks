
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

def test_half_precision_input():
    # This test feeds a half-precision tensor to the CUDA kernel.
    # The kernel code dispatches only for float and double.
    # Thus, passing a half tensor should raise an error.
    mod = build_kernel()
    x = torch.randn(32, 32, dtype=torch.half, device="cuda")
    with pytest.raises(RuntimeError):
        # The kernel should fail to dispatch for half precision.
        _ = mod.forward(x)

def test_non_contiguous_input():
    # This test creates a non-contiguous tensor by transposing a contiguous tensor.
    # The kernel assumes contiguous memory; hence, the output of the customized kernel
    # will be computed on incorrect memory addresses and will not match torch.selu.
    mod = build_kernel()
    # Create a contiguous tensor and then make it non-contiguous.
    x = torch.randn(64, 128, device="cuda", dtype=torch.float32)
    x_non_contig = x.t()  # Transpose: non-contiguous view.
    
    # Compute SELU using the custom CUDA kernel and PyTorch's implementation.
    y_kernel = mod.forward(x_non_contig)
    y_torch = torch.selu(x_non_contig)
    
    # Expect the outputs to not match because the kernel incorrectly assumes contiguity.
    # The test passes if there is a significant difference.
    if torch.allclose(y_kernel, y_torch, atol=1e-5):
        pytest.fail("Kernel did not exhibit error with non-contiguous input as expected.")
