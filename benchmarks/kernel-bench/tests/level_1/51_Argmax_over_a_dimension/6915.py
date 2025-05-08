
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

# Issue 1:
# Test that non-float32 tensors are rejected.
def test_non_float32_input():
    my_module = build_kernel()
    # Create a tensor of type double
    x = torch.randn(16, 256, 256, dtype=torch.double, device="cuda")
    with pytest.raises(RuntimeError, match="Only float32 is supported."):
        # The C++ TORCH_CHECK should trigger an error.
        my_module.forward(x, 1)

# Issue 2:
# Test the behavior when the reduction dimension is empty.
# For a tensor with size 0 in the reduction dimension, torch.argmax raises an error.
def test_empty_reduction_dimension():
    my_module = build_kernel()
    # Create a tensor with shape [2, 0, 3] and set reduction dimension to 1.
    x = torch.randn(2, 0, 3, dtype=torch.float32, device="cuda")
    # The PyTorch torch.argmax usually throws a runtime error on empty tensors.
    # However, our custom kernel does not check for this and will return 0.
    out_kernel = my_module.forward(x, 1)
    # We expect the output shape to be [2,3] per our output shape construction.
    # Check that the returned indices are all 0, which is the (incorrect) fallback value.
    expected = torch.zeros((2, 3), dtype=torch.long, device="cuda")
    assert torch.equal(out_kernel, expected), (
        "Kernel did not handle empty reduction dimension correctly; expected all zeros, "
        f"got {out_kernel}"
    )

# Issue 3:
# Test that the kernel handles cases when the reduction dimension is very large.
# Although the kernel is designed to use one warp per (outer, inner) pair, its design
# may limit its flexibility. Here we compare the output of the kernel to torch.argmax
# on a tensor with a large reduction dimension (and where the number of iterations per thread > 1).
def test_large_reduction_dimension():
    my_module = build_kernel()
    # Construct a tensor where the reduction dimension is larger than 32.
    batch_size = 4
    outer = 3
    reduction = 100  # > 32 so each warp thread will have to iterate more than once.
    inner = 5
    # Tensor shape: [batch_size, outer, reduction, inner]
    x = torch.randn(batch_size, outer, reduction, inner, dtype=torch.float32, device="cuda")
    # Pick a reduction dimension. The kernel expects a single dimension reduction.
    # We choose dim=2 (the reduction dimension).
    out_kernel = my_module.forward(x, 2)
    
    # Get the reference result using torch.argmax. Note that torch.argmax returns a tensor
    # with the reduced dimension removed.
    out_ref = torch.argmax(x, dim=2)
    
    # If the kernel were more general, these results should match.
    assert torch.equal(out_kernel, out_ref), (
        "Kernel output does not match torch.argmax output for a large reduction dimension."
    )
