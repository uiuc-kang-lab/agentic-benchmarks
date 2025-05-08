
import pytest
import torch
from torch.utils.cpp_extension import load

# Build the CUDA extension from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="gelu_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Passing a non-contiguous tensor should trigger issues related to improper alignment.
def test_non_contiguous_input():
    my_module = build_kernel()
    # Create a contiguous tensor then make it non-contiguous by transposing.
    x = torch.randn(16, 16384, device="cuda", dtype=torch.float32).t()  # now shape (16384, 16)
    # Notice: our kernel expects a contiguous float tensor.
    with pytest.raises(RuntimeError):
        # The wrong alignment / non-contiguity may trigger undefined behavior. In many cases, this will result in a launch error.
        y = my_module.forward(x)
        # To force synchronization and check for errors.
        torch.cuda.synchronize()

# Test 2: Passing a tensor with a number of elements that is not divisible by 4.
# Although the remainder kernel attempts to cover leftover elements, its indexing is error‐prone.
def test_remainder_indexing():
    my_module = build_kernel()
    # Create a tensor with numel not divisible by 4.
    # For instance, make a 1-D tensor with 6 elements.
    x = torch.randn(6, device="cuda", dtype=torch.float32)
    y = my_module.forward(x)
    torch.cuda.synchronize()
    # Compare with PyTorch's reference GELU.
    y_ref = torch.nn.functional.gelu(x)
    # The unusual indexing in the remainder branch may lead to mismatches.
    assert torch.allclose(y, y_ref, atol=1e-5), "Mismatch detected in remainder kernel handling."

# Test 3: Passing a tensor of unsupported type (e.g. float64) should raise an error.
def test_unsupported_dtype():
    my_module = build_kernel()
    x = torch.randn(16, 16384, device="cuda", dtype=torch.float64)
    with pytest.raises(RuntimeError):
        y = my_module.forward(x)
        torch.cuda.synchronize()
