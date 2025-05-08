
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build the kernel from the given kernel.cu file
def build_kernel():
    module = load(
        name="my_cuda_kernel",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return module

@pytest.fixture(scope="module")
def cuda_module():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return build_kernel()

# Test case 1: Trigger the issue of using fmaxf with double precision inputs.
def test_double_input_issue(cuda_module):
    # Create a double precision tensor (float64) which is not correctly handled by fmaxf.
    # The kernel dispatched via AT_DISPATCH_FLOATING_TYPES_AND_HALF will run with scalar_t set to double,
    # but the fmaxf call inside the kernel is not valid for doubles.
    batch_size, dim1, dim2 = 8, 32, 16
    x = torch.randn(batch_size, dim1, dim2, dtype=torch.double, device='cuda')
    # Get reference result using torch.max
    ref = torch.max(x, dim=1)[0]
    # Run the custom kernel reduction along dimension 1.
    out = cuda_module.forward(x, 1)
    torch.cuda.synchronize()
    # The output is likely to be incorrect because of the fmaxf usage.
    # We expect a disagreement.
    assert not torch.allclose(out, ref, atol=1e-5), (
        "Test failed: The kernel unexpectedly produced correct results for double precision input. "
        "This indicates that the type-specific max function is not properly dispatched."
    )

# Test case 2: Trigger the issue with non-contiguous input.
def test_non_contiguous_input_issue(cuda_module):
    # Create a contiguous tensor and then create a non-contiguous view (e.g. via transpose).
    batch_size, dim1, dim2 = 8, 32, 16
    x = torch.randn(batch_size, dim1, dim2, device='cuda')
    # Make a non-contiguous tensor by transposing two dimensions that are not the reduction dimension.
    # For example, if reduction is along dim1, transpose dim0 and dim1.
    x_noncontig = x.transpose(0, 1)
    # The kernel expects a contiguous layout with specific assumptions.
    # Get the reference (forcing contiguous before reduction)
    ref = torch.max(x_noncontig.contiguous(), dim=1)[0]
    # Run the custom kernel reduction along dimension 1 on the non-contiguous tensor.
    out = cuda_module.forward(x_noncontig, 1)
    torch.cuda.synchronize()
    # The result will likely be incorrect due to the layout assumptions.
    assert not torch.allclose(out, ref, atol=1e-5), (
        "Test failed: The kernel unexpectedly produced correct results for a non-contiguous input. "
        "This indicates that the kernel does not correctly handle non-contiguous tensor layouts."
    )

# Test case 3: Check absence of error checking after kernel launch.
# This test indirectly checks that errors from the kernel launch are not caught.
# We create a scenario where the kernel launch should fail due to invalid parameters.
def test_error_checking_issue(cuda_module):
    # Intentionally trigger an error by passing an invalid reduction dimension.
    batch_size, dim1, dim2 = 8, 32, 16
    x = torch.randn(batch_size, dim1, dim2, device='cuda')
    invalid_dim = 100  # Clearly out of bounds.
    with pytest.raises(RuntimeError):
        # Even though the kernel code does not explicitly check for this, the launch might
        # result in accessing out-of-bound memory causing a CUDA error.
        cuda_module.forward(x, invalid_dim)
        torch.cuda.synchronize()
