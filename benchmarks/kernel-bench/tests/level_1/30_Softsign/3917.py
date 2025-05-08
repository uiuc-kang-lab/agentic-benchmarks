
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="softsign_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Kernel does not support non-float32 tensors.
def test_non_float_input():
    my_module = build_kernel()
    # Create a double tensor (float64) rather than float32.
    x = torch.randn(1024, device="cuda", dtype=torch.float64)
    with pytest.raises(RuntimeError):
        # The kernel always casts the pointer as float*, so this should lead to memory errors.
        out = my_module.forward(x)
        torch.cuda.synchronize()

# Issue 2: Missing kernel launch error checking.
# We simulate a situation that would likely generate a kernel runtime error.
# Here, we create an input tensor with an incorrect memory layout by making it non-contiguous.
def test_non_contiguous_input():
    my_module = build_kernel()
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.float32).t()  # Transpose makes it non-contiguous.
    with pytest.raises(RuntimeError) as excinfo:
        out = my_module.forward(x)
        torch.cuda.synchronize()
    assert "contiguous" in str(excinfo.value), "Expected an error about non-contiguous tensors."

# Issue 3: Lack of a backward kernel means the custom op is not autograd-differentiable.
def test_backward_not_implemented():
    my_module = build_kernel()
    # Create a float tensor that requires grad.
    x = torch.randn(1024, device="cuda", dtype=torch.float32, requires_grad=True)
    out = my_module.forward(x)
    # Try to perform backward propagation.
    with pytest.raises(RuntimeError) as excinfo:
        out.sum().backward()
    assert "not implemented" in str(excinfo.value).lower() or "backward" in str(excinfo.value).lower(), \
        "Expected an error indicating that the backward pass is not implemented."
