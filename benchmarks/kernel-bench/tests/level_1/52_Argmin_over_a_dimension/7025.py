
import pytest
import torch
from torch.utils.cpp_extension import load
import os

def build_kernel():
    # Rebuild the CUDA extension from kernel.cu file
    base_path = os.path.dirname(os.path.abspath(__file__))
    module = load(
        name="argmin_kernel_module",
        sources=[os.path.join(base_path, "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return module

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_integer_dtype_issue():
    """
    Test the kernel with an integer tensor.
    The kernel uses INFINITY for initialization which is invalid for integer types.
    Expected behavior: either an error is raised or the result is incorrect.
    """
    kernel_module = build_kernel()
    # Create an integer tensor, e.g., int32.
    # Note: torch.argmin supports integer tensors, but our kernel initialization
    # using INFINITY will be problematic.
    batch, dim1, dim2 = 8, 32, 32
    x = torch.randint(0, 100, (batch, dim1, dim2), dtype=torch.int32, device="cuda")
    # Our kernel uses a reduction dimension that we control.
    try:
        result = kernel_module.forward(x, 1)
    except Exception as e:
        pytest.fail(f"Kernel raised an exception for integer type input: {e}")
    # Compare with torch.argmin (which is correct) and intentionally expect a discrepancy.
    expected = torch.argmin(x, dim=1)
    # The kernel is likely to produce wrong results because of NAN/INFINITY usage.
    # We assert that the kernel output does not match the reference.
    assert not torch.equal(result, expected), "Kernel unexpectedly returned correct result for integer dtype input."

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_noncontiguous_tensor_issue():
    """
    Test the kernel with a non-contiguous tensor.
    The kernel computes slice pointer arithmetic assuming contiguous layout.
    For non-contiguous tensors the result should be incorrect.
    """
    kernel_module = build_kernel()
    # Create a contiguous tensor and then transpose it to make non-contiguous memory layout.
    batch, dim1, dim2 = 8, 64, 64
    x = torch.randn(batch, dim1, dim2, device="cuda")
    # Transpose to force non-contiguity. Transpose between dim0 and dim1.
    x_noncontig = x.transpose(0, 1)
    # Invoke the kernel along dimension 1 (which corresponds to the original dim0 now)
    try:
        result = kernel_module.forward(x_noncontig, 1)
    except Exception as e:
        pytest.fail(f"Kernel raised an exception for non-contiguous input: {e}")
    expected = torch.argmin(x_noncontig, dim=1)
    # Because the kernel does not account for non-contiguous layout correctly,
    # the output is expected to be different from torch.argmin.
    assert not torch.equal(result, expected), "Kernel unexpectedly returned correct result for non-contiguous input."
