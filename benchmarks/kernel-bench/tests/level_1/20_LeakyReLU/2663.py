
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="leaky_relu_kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

def torch_leaky_relu(x, negative_slope):
    # Reference implementation using PyTorch's built-in function.
    return torch.nn.functional.leaky_relu(x, negative_slope=negative_slope)

# Issue 1 Test: Passing a tensor that is not float32 should result in incorrect behavior.
def test_invalid_dtype():
    my_module = build_kernel()
    negative_slope = 0.01
    # Create an input tensor with double precision (float64)
    x = torch.randn(1024, device="cuda", dtype=torch.float64)
    # The CHECK_INPUT macro in the kernel does not check dtype,
    # so the kernel will run (misinterpreting the memory).
    out = my_module.forward(x, negative_slope)
    # Compute reference output by explicitly converting
    x_float = x.float()
    out_ref = torch_leaky_relu(x_float, negative_slope)
    # Because of the dtype mismatch, the output is expected NOT to match
    # (the kernel wrongly treats the double data as float data).
    with pytest.raises(AssertionError):
        # We expect the values to not be almost equal.
        torch.testing.assert_allclose(out, out_ref, atol=1e-5)

# Issue 2 Test: If an error occurs during kernel execution, the lack of error checking
# may cause silent failures. We simulate this scenario by creating a non-contiguous tensor.
def test_non_contiguous_tensor():
    my_module = build_kernel()
    negative_slope = 0.01
    x = torch.randn(1024, device="cuda", dtype=torch.float32)
    # Make x non-contiguous by transposing a 2D tensor.
    x = x.view(32, -1).t()
    # The kernel CHECK_INPUT macro expects a contiguous tensor,
    # so this should trigger an error.
    with pytest.raises(RuntimeError, match="must be contiguous"):
        my_module.forward(x, negative_slope)
        
if __name__ == "__main__":
    pytest.main([__file__])
