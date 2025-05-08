
import pytest
import torch
from torch.utils.cpp_extension import load

# This helper function compiles the CUDA extension.
def build_kernel():
    cuda_module = load(
        name="gelu_cuda_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Pass an input tensor with a non-float32 type (e.g. float64) to trigger the type assumption issue.
def test_input_tensor_wrong_type():
    my_module = build_kernel()
    # Create a tensor with dtype float64 instead of float32.
    x = torch.randn(100, device="cuda", dtype=torch.float64)
    with pytest.raises(RuntimeError, match="Input tensor must be on CUDA|"):
        # Note: TORCH_CHECK in the C++ extension doesn't check type; however, using wrong type will
        # lead to invalid memory interpretation or crash. Here we use pytest.raises to expect an error.
        out = my_module.forward(x)
        torch.cuda.synchronize()

# Test 2: Pass a non-contiguous tensor to trigger the contiguity check.
def test_input_tensor_not_contiguous():
    my_module = build_kernel()
    # Create a contiguous tensor then take a non-contiguous slice.
    x_full = torch.randn(100, 100, device="cuda", dtype=torch.float32)
    x = x_full.t()  # Transpose makes it non-contiguous in PyTorch 1.x
    with pytest.raises(RuntimeError, match="Input tensor must be contiguous"):
        out = my_module.forward(x)
        torch.cuda.synchronize()
