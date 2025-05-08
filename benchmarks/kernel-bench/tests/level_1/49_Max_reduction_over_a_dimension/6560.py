
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the CUDA extension from kernel.cu
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

def cuda_max_reduce(module, input_tensor, dim):
    # Call the CUDA kernel wrapped by the module
    return module.forward(input_tensor, dim)

# Issue 1: Non-contiguous Tensor
def test_non_contiguous_tensor():
    # Create a contiguous tensor and then make it non-contiguous via transpose
    x = torch.randn(16, 256, 256, device="cuda")
    x_nc = x.transpose(1, 2)  # now non-contiguous, and the reduction dimension is no longer arranged as assumed.
    
    cuda_module = build_kernel()
    # We'll reduce along dim=1; for non-contiguous tensor, the kernel arithmetic is wrong.
    try:
        out_cuda = cuda_max_reduce(cuda_module, x_nc, 1)
    except Exception as e:
        pytest.skip("Kernel did not run on non-contiguous input (expected behavior).")
    # Get reference using torch.max along the same user-provided dim.
    out_ref = torch.max(x_nc, dim=1)[0]
    # The outputs will not match due to wrong indexing.
    assert not torch.allclose(out_cuda, out_ref), \
        "For a non-contiguous tensor, the CUDA kernel returned the same result as torch.max, " \
        "which is unexpected given its assumptions about memory layout."

# Issue 2: Unsupported (Integer) Data Types
def test_integer_dtype_not_supported():
    # Create a tensor with an integer type (e.g., int32)
    x_int = torch.randint(low=0, high=100, size=(16, 256, 256), device="cuda", dtype=torch.int32)
    cuda_module = build_kernel()
    # Expect the extension not to support int types: it should throw an error.
    with pytest.raises(RuntimeError):
        _ = cuda_max_reduce(cuda_module, x_int, 1)

# Issue 3: Inadequate handling of __half precision (float16)
def test_half_precision_handling():
    # Create a half precision tensor.
    x_half = torch.randn(16, 256, 256, device="cuda", dtype=torch.float16)
    cuda_module = build_kernel()
    try:
        out_cuda = cuda_max_reduce(cuda_module, x_half, 1)
    except Exception as e:
        # If an error is raised, then the kernel does not support half properly.
        pytest.skip("Kernel raised an exception for half precision input, indicating inadequate handling.")
    
    # Compute reference using torch.max; if the kernel uses an improper max operator,
    # the results may differ.
    out_ref = torch.max(x_half, dim=1)[0]
    
    # Check that outputs differ, which indicates an issue in half handling.
    if torch.allclose(out_cuda, out_ref, atol=1e-3):
        pytest.fail("Expected discrepancy in half precision results due to improper handling of __half type.")
