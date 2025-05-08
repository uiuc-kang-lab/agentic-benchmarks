
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="min_reduce_kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Lack of error checking for asynchronous kernel launches.
# We simulate a potential kernel launch error by using an unsupported tensor type.
def test_bool_input():
    # AT_DISPATCH_ALL_TYPES does not include bool, so this should trigger an error.
    kernel_module = build_kernel()
    # Create a bool tensor on CUDA. The kernel isn’t defined for bool.
    input_tensor = torch.zeros((4, 3, 2), dtype=torch.bool, device="cuda")
    with pytest.raises(RuntimeError) as err:
        kernel_module.forward(input_tensor, 1)
    # The error message should indicate that the scalar type wasn’t dispatched.
    assert "AT_DISPATCH" in str(err.value) or "not implemented" in str(err.value)

# Issue 2: No validation for an empty reduction dimension.
def test_empty_reduction():
    kernel_module = build_kernel()
    # Create an input tensor where the reduction dimension has size 0.
    # For example, suppose we reduce over dimension 1.
    input_tensor = torch.randn((4, 0, 5), device="cuda", dtype=torch.float32)
    # Although the PyTorch torch.min would normally throw an error with an empty dim,
    # our kernel does not perform any check and will likely access out-of-bounds memory.
    with pytest.raises(RuntimeError) as err:
        kernel_module.forward(input_tensor, 1)
    # The error message may be a CUDA error (e.g. illegal memory access).
    assert "illegal" in str(err.value).lower() or "out of bounds" in str(err.value).lower()

# Issue 3: No check on CUDA stream creation.
# We simulate an error in stream creation indirectly by passing an invalid dimension index.
def test_invalid_dim():
    kernel_module = build_kernel()
    # Create a valid CUDA tensor.
    input_tensor = torch.randn((4, 10, 5), device="cuda", dtype=torch.float32)
    # Pass an invalid reduction dimension to trigger the TORCH_CHECK error.
    with pytest.raises(RuntimeError) as err:
        # dim=3 is out of range for a 3D tensor (indices: 0,1,2)
        kernel_module.forward(input_tensor, 3)
    assert "dim out of range" in str(err.value)

# Additional test to ensure that non-CUDA inputs are rejected.
def test_cpu_input():
    kernel_module = build_kernel()
    input_tensor = torch.randn((4, 3, 2), device="cpu", dtype=torch.float32)
    with pytest.raises(RuntimeError) as err:
        kernel_module.forward(input_tensor, 1)
    assert "input must be a CUDA tensor" in str(err.value)
