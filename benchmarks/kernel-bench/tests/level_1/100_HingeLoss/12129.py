
import pytest
import torch
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

def test_incorrect_loop_increment():
    # This test creates an input tensor large enough that the grid has multiple blocks.
    # Due to the wrong loop increment in the kernel (i += blockDim.x instead of i += stride),
    # some elements will not be processed and the computed mean will be wrong.
    N = 1024  # Enough elements to guarantee more than one block (threads per block = 256)
    predictions = torch.randn(N, device="cuda", dtype=torch.float32)
    targets = (torch.randint(0, 2, (N,), device="cuda", dtype=torch.float32) * 2) - 1

    kernel = build_kernel()
    result = kernel.forward(predictions, targets)
    # Compute the expected hinge loss using PyTorch
    expected = torch.mean(torch.clamp(1 - predictions * targets, min=0))

    # Because of the incorrect loop increment, the kernel should not compute the full result.
    # We expect a significant difference.
    assert not torch.allclose(result, expected, atol=1e-5), \
        f"Kernel computed the same result as reference, but it should have skipped some elements."

def test_dtype_not_float32():
    # This test passes double (float64) tensors to the kernel.
    # Since the kernel assumes float32 and does not check the type,
    # it will use the wrong data pointer type leading to erroneous results.
    N = 256
    predictions = torch.randn(N, device="cuda", dtype=torch.float64)
    # Create targets in float64: map randint{0,1} to {-1,1}
    targets = (torch.randint(0, 2, (N,), device="cuda", dtype=torch.int64).to(torch.float64) * 2) - 1

    kernel = build_kernel()
    result = kernel.forward(predictions, targets)
    expected = torch.mean(torch.clamp(1 - predictions * targets, min=0))
    
    # The result is likely wrong due to the type mismatch.
    assert not torch.allclose(result.to(torch.float64), expected, atol=1e-5), \
        "Kernel should not support float64 inputs correctly."

def test_non_contiguous_input():
    # This test provides non-contiguous tensors to the kernel.
    # The CHECK_CONTIGUOUS macro should cause an error.
    N = 256
    base_predictions = torch.randn(N * 2, device="cuda", dtype=torch.float32)
    # Slicing with a step will produce a non-contiguous tensor.
    predictions = base_predictions[::2]
    base_targets = torch.randint(0, 2, (N * 2,), device="cuda", dtype=torch.float32)
    targets = (base_targets[::2] * 2) - 1

    kernel = build_kernel()
    with pytest.raises(RuntimeError, match="must be contiguous"):
        kernel.forward(predictions, targets)
