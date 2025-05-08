
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the CUDA extension from the kernel.cu file.
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_shared_memory_type_mismatch():
    """
    Issue 1:
    Using double precision input may lead to shared memory problems because the shared mem size is computed with sizeof(float).
    This test checks whether the output deviates from torch.mean.
    """
    cuda_module = build_kernel()
    # Create a tensor with double precision.
    x = torch.randn(16, 256, 256, dtype=torch.double, device='cuda')
    # Reduce along dimension 1.
    y_cuda = cuda_module.forward(x, 1)
    torch.cuda.synchronize()
    y_ref = torch.mean(x, dim=1)
    # We expect significant deviation because shared memory size was computed using float.
    assert not torch.allclose(y_cuda, y_ref, atol=1e-6), "Kernel output unexpectedly equals reference output for double precision."

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_non_contiguous_input():
    """
    Issue 2:
    The kernel assumes a contiguous memory layout. This test creates a non-contiguous input tensor
    by transposing two dimensions. The kernel’s result will likely differ from torch.mean.
    """
    cuda_module = build_kernel()
    # Create a contiguous tensor and then transpose to get a non-contiguous layout.
    x = torch.randn(16, 256, 256, device='cuda')
    # Transpose such that the reduced dimension is not contiguous.
    x_t = x.transpose(1, 2)  # Now reduction along dimension 1 is not contiguous.
    y_cuda = cuda_module.forward(x_t, 1)
    torch.cuda.synchronize()
    y_ref = torch.mean(x_t, dim=1)
    # The results will differ because the kernel's indexing assumes contiguity.
    assert not torch.allclose(y_cuda, y_ref, atol=1e-5), "Kernel output unexpectedly equals reference output for non-contiguous input."

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_half_precision_unsupported():
    """
    Issue 3:
    The kernel dispatch macro AT_DISPATCH_FLOATING_TYPES does not include half precision.
    This test checks that using half precision input raises an error.
    """
    cuda_module = build_kernel()
    x = torch.randn(16, 256, 256, device='cuda', dtype=torch.half)
    with pytest.raises(RuntimeError):
        # Expecting an error because half precision is not supported by AT_DISPATCH_FLOATING_TYPES.
        _ = cuda_module.forward(x, 1)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_kernel_launch_error_checking():
    """
    Issue 4:
    Since there is no error checking after the kernel launch, a problematic input (e.g., an empty reduction dimension)
    might lead to a silent error or undefined behavior. Here we try with an empty tensor along the reduction dimension.
    """
    cuda_module = build_kernel()
    # Create a tensor with an empty dimension to reduce.
    x = torch.randn(16, 0, 256, device='cuda')
    # When reducing over dim=1 (which is empty) the division by zero will occur.
    # torch.mean would normally produce NaNs for empty reduction.
    y_cuda = cuda_module.forward(x, 1)
    torch.cuda.synchronize()
    y_ref = torch.mean(x, dim=1)
    # Check if the outputs contain NaN (or are significantly different).
    assert torch.isnan(y_cuda).any() or not torch.allclose(y_cuda, y_ref, equal_nan=True), \
        "Kernel did not propagate error for empty reduction dimension as expected."
