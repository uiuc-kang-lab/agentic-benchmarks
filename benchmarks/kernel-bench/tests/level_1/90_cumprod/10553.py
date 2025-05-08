
import pytest
import torch
from torch.utils.cpp_extension import load

# Function to build and load the CUDA extension from 'kernel.cu'.
def build_kernel():
    cuda_module = load(
        name="cumprod_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Non‐contiguous input leads to wrong results.
def test_non_contiguous_input():
    # Create a contiguous tensor and then transpose to get a non‐contiguous layout.
    # We compute cumprod along dim=0 (first dimension) of a transposed tensor.
    A = torch.randn(128, 4000, device="cuda")
    A_t = A.t()  # Now shape (4000, 128); note that cumprod along dim=0 in A_t is over non‐contiguous data.
    # Our kernel expects a contiguous pattern along the chosen dimension.
    # Use dim=0 so that the stride of that dimension is not 1.
    my_module = build_kernel()
    # Get result from our CUDA kernel.
    result_kernel = my_module.forward(A_t, 0)
    # Compute reference using torch.cumprod on the same (non‐contiguous) tensor.
    result_ref = torch.cumprod(A_t, dim=0)
    # The results should be close if the kernel handled non‐contiguous input.
    # We expect them NOT to match because of the wrong handling of strides.
    assert not torch.allclose(result_kernel, result_ref, atol=1e-5), (
        "Test failed to trigger non-contiguous input issue: kernel result unexpectedly matches torch.cumprod."
    )

# Issue 2: Unsupported input type (integer) should trigger an error.
def test_unsupported_integer_type():
    # Create a tensor of integer type.
    A_int = torch.randint(1, 10, (128, 4000), device="cuda", dtype=torch.int32)
    my_module = build_kernel()
    with pytest.raises(RuntimeError):
        # Expect the extension to raise an error because int32 is not among the dispatched types.
        _ = my_module.forward(A_int, 1)

# Issue 3: Lack of kernel launch error-checking.
# One common way to trigger a kernel launch error is to use an invalid grid configuration.
# Here, we simulate this by passing a bad dimension value (e.g. a dimension
# that is out of bounds of the tensor shape) to force an out-of-range memory access.
def test_invalid_dimension():
    A = torch.randn(128, 4000, device="cuda")
    my_module = build_kernel()
    # Using a dimension that does not exist should lead to wrong behaviour or a runtime failure.
    # Since our kernel does not do explicit error checking, the error might surface as a runtime crash.
    # We wrap the call in pytest.raises to catch the error.
    with pytest.raises(Exception):
        _ = my_module.forward(A, 5)

if __name__ == "__main__":
    pytest.main([__file__])
