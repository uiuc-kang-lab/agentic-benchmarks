
import torch
import pytest
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

# Issue 1: Non-contiguous tensor input.
# The kernel assumes a contiguous layout; if a non-contiguous tensor is passed, the computed
# cumulative product will be incorrect.
def test_non_contiguous_input():
    my_module = build_kernel()
    # Create a tensor and then transpose it to break contiguity.
    x = torch.randn(128, 4000, device="cuda")
    x = x.t()  # now shape is (4000, 128) and non-contiguous for the intended cumulative dim.
    # We intend to perform cumprod along dim=1 (the original second dimension)
    # but after transpose, dim=1 corresponds to original dim=0.
    try:
        y_kernel = my_module.forward(x, 1)  # pass cumulative dim = 1
    except Exception as e:
        pytest.skip("Kernel did not support non-contiguous inputs (as expected).")
    # Compute reference cumulative product on the original tensor using torch.cumprod
    y_ref = torch.cumprod(x, dim=1)
    # In case the kernel did run, check that the result is not matching torch.cumprod.
    # (If the kernel were generalized, it would match.)
    assert not torch.allclose(y_kernel, y_ref, atol=1e-5), (
        "Kernel unexpectedly produced correct results on a non-contiguous input."
    )

# Issue 2: Higher dimensional tensor with an arbitrary cumulative product dimension.
# The current indexing logic only works for a 2D contiguous layout.
def test_higher_dimensional_tensor():
    my_module = build_kernel()
    # Create a 3D tensor and perform cumprod along a non-last dimension.
    x = torch.randn(16, 32, 64, device="cuda")
    dim = 1  # perform cumprod along the second dimension
    # Because the kernel computes total_batches as numel()/dim_size, its batch decomposition
    # is not correct for a 3D tensor.
    y_kernel = my_module.forward(x, dim)
    y_ref = torch.cumprod(x, dim)
    # We expect the kernel to fail producing correct output.
    assert not torch.allclose(y_kernel, y_ref, atol=1e-5), (
        "Kernel unexpectedly produced correct results on a higher dimensional tensor."
    )

# Issue 3: Lack of kernel launch error checking.
# While the kernel might run and produce an output in simple cases,
# we can try to trigger a situation (e.g. by using an extremely large tensor)
# in which the kernel launch might fail. Without error checking, this error will go unnoticed.
def test_extremely_large_tensor():
    my_module = build_kernel()
    # Create a tensor that is very large so that the kernel launch parameters may be invalid.
    try:
        # Note: Adjust the shape depending on available device memory.
        x = torch.randn(10000, 10000, device="cuda")
        y_kernel = my_module.forward(x, 1)
        # If kernel launch errors occur, they might be delayed till synchronization.
        torch.cuda.synchronize()
        # If no error is thrown, then the kernel is unexpectedly handling this extreme case.
        pytest.skip("Kernel handled extremely large tensor without error checking unexpectedly.")
    except RuntimeError as e:
        # An error is expected due to resource exhaustion or kernel configuration error.
        assert "launch" in str(e) or "kernel" in str(e), "Unexpected error when launching kernel."

if __name__ == "__main__":
    pytest.main([__file__])
