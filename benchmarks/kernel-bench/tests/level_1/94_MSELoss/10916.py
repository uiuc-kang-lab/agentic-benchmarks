
import pytest
import torch
from torch.utils.cpp_extension import load


def build_kernel():
    cuda_module = load(
        name="kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tail_elements_bug():
    # Test case designed to expose the tail handling issue.
    # We create tensors with a number of elements that is not a multiple of WARP_SIZE (32)
    # so that many tail elements might be skipped.
    # For example, using 37 elements (which is > 1 and < 32 tail elements)
    n = 37
    predictions = torch.randn(n, device="cuda", dtype=torch.float32)
    targets = torch.randn(n, device="cuda", dtype=torch.float32)
  
    # Compute reference MSE using PyTorch
    mse_ref = torch.mean((predictions - targets) ** 2)
  
    # Compute MSE using our custom CUDA kernel
    kernel_mod = build_kernel()
    mse_kernel = kernel_mod.forward(predictions, targets)

    # In a correct implementation the results should be nearly identical.
    # If the tail processing bug is triggered, the result will be noticeably off.
    torch.cuda.synchronize()
    assert not torch.allclose(mse_kernel, mse_ref, atol=1e-5), (
        "Test failed: Kernel tail-processing bug not detected. "
        "Results match PyTorch even though the kernel appears to have an error."
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_incorrect_grid_stride_accumulation():
    # Test case designed to trigger the bug when each thread is expected
    # to iterate over multiple chunks. For a large number of elements and a grid_size
    # forced to be a multiple of 32, grid_stride might be overly large causing threads
    # to process only a subset of the intended elements.
    n = 10**6 + 17  # deliberately not a multiple of BLOCK_SIZE or warp size
    predictions = torch.randn(n, device="cuda", dtype=torch.float32)
    targets = torch.randn(n, device="cuda", dtype=torch.float32)
  
    # Compute reference MSE using PyTorch
    mse_ref = torch.mean((predictions - targets) ** 2)
  
    # Compute MSE using our custom CUDA kernel
    kernel_mod = build_kernel()
    mse_kernel = kernel_mod.forward(predictions, targets)

    torch.cuda.synchronize()
    # Because of missing accumulation in the grid-stride loop the kernel result
    # will be significantly different from the reference.
    assert not torch.allclose(mse_kernel, mse_ref, atol=1e-5), (
        "Test failed: Kernel grid-stride bug not detected. "
        "Kernel result matches reference despite expected bug."
    )
