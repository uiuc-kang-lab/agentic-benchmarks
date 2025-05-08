
import pytest
import torch
import math
from torch.utils.cpp_extension import load
from torch.nn.functional import leaky_relu

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Only supports float32 (data type issue)
def test_non_float_dtype():
    # Create a double tensor instead of float32
    x = torch.randn(1024, device="cuda", dtype=torch.double)
    # Load our kernel extension
    module = build_kernel()
    # The kernel blindly uses data_ptr<float>(), so the result will be misinterpreted.
    out = module.forward(x, 0.01)
    # Compute reference using built-in leaky_relu on float32 tensor conversion.
    x_float = x.float()
    out_ref = leaky_relu(x_float, negative_slope=0.01)
    # Because of data type mismatch, the result should differ significantly.
    diff = (out.float() - out_ref).abs().max().item()
    assert diff > 1e-3, f"Kernel unexpectedly produced correct results for double tensor (diff={diff})."

# Issue 2: Using custom streams instead of PyTorch current stream may cause synchronization issues.
# Here we simulate a situation where an operation on a noncontiguous tensor
# is passed; since the kernel enforces contiguous inputs (via CHECK_CONTIGUOUS),
# this triggers an error.
def test_non_contiguous_input():
    x = torch.randn(64, 32, device="cuda", dtype=torch.float32).t()  # transpose to make it non-contiguous
    module = build_kernel()
    with pytest.raises(RuntimeError, match="must be contiguous"):
        module.forward(x, 0.01)

# Issue 3: Lack of proper CUDA error checking after kernel launch.
# We simulate an erroneous situation by using a tensor with very few elements.
# This exposes potential out-of-bound or kernel launch issues due to incorrect error checking.
def test_small_tensor_boundary():
    # A tensor where n is very small, forcing at least one chunk to be nearly empty,
    # testing boundary conditions in stream-based partitioning.
    x = torch.randn(2, device="cuda", dtype=torch.float32)
    module = build_kernel()
    out = module.forward(x, 0.01)
    out_ref = leaky_relu(x, negative_slope=0.01)
    # Even if error checking is missing, the output should nonetheless be correct in this simple case.
    assert torch.allclose(out, out_ref, atol=1e-5), "Kernel output differs from reference output for small tensor."

