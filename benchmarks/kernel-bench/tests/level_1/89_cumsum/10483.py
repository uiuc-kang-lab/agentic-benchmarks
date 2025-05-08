
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Test case 1: Non-float32 input triggering type incompatibility.
def test_non_float_tensor():
    my_module = build_kernel()
    # Create a double tensor. The kernel uses data_ptr<float>() so it will misinterpret the data.
    x = torch.randn(128, 4000, dtype=torch.double, device="cuda")
    # We expect that the result will be incorrect.
    out = my_module.forward(x, 1)
    torch.cuda.synchronize()
    # Compute the correct cumsum cast to double for reference.
    expected = torch.cumsum(x, dim=1)
    # Check that the kernel output is not allclose to the expected result,
    # indicating that the kernel did not handle a non-float32 type correctly.
    assert not torch.allclose(out, expected, atol=1e-5), \
        "Kernel unexpectedly produced correct output for a non-float32 tensor."

# Test case 2: Non-contiguous input triggering the contiguous check.
def test_non_contiguous_tensor():
    my_module = build_kernel()
    # Create a contiguous tensor first.
    x = torch.randn(128, 4000, dtype=torch.float32, device="cuda")
    # Make the tensor non-contiguous by transposing (if possible)
    # Note: For a 2D tensor, transpose returns a non-contiguous tensor.
    x_noncontig = x.t()
    # When calling the kernel with a non-contiguous tensor, the CHECK_CONTIGUOUS macro should trigger an error.
    with pytest.raises(RuntimeError, match="must be contiguous"):
        _ = my_module.forward(x_noncontig, 0)
        torch.cuda.synchronize()

# Test case 3: Empty cumulative dimension (stride = 0) leading to uninitialized output.
def test_empty_cumsum_dimension():
    my_module = build_kernel()
    # Create a tensor with an empty cumulative dimension. For example, dim=1 is empty.
    x = torch.empty(128, 0, dtype=torch.float32, device="cuda")
    out = my_module.forward(x, 1)
    torch.cuda.synchronize()
    # The expected output for an empty cumulative dimension is an empty tensor.
    expected = torch.cumsum(x, dim=1)
    # Since the kernel does not properly handle this case, its output will not match the expected result.
    assert not torch.equal(out, expected), \
        "Kernel produced correct output on an empty cumulative dimension, but it should not."

