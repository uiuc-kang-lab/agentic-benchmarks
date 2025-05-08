
import pytest
import torch
from torch.utils.cpp_extension import load
import numpy as np

def build_kernel():
    # Build the CUDA extension from the provided kernel.cu file.
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_issue_uninitialized_shared_memory_constant():
    # Issue 1: When num_features is much smaller than THREADS_PER_BLOCK,
    # some threads do not contribute and shared memory remains uninitialized,
    # causing an incorrect softmax result.
    # Use a tensor that uses the constant memory kernel (total elements <= MAX_CONST_INPUT_SIZE)
    # with a small number of features.
    batch_size = 2
    num_features = 128  # less than THREADS_PER_BLOCK (256)
    x = torch.randn(batch_size, num_features, device="cuda", dtype=torch.float32)
    # Expected output using PyTorch softmax
    expected = torch.softmax(x, dim=1)
    
    module = build_kernel()
    y = module.forward(x)
    torch.cuda.synchronize()
    # The bug should cause an incorrect result.
    # We test that the computed output does NOT match PyTorch's softmax.
    # (If the kernel were correct, these would match.)
    assert not torch.allclose(y, expected, atol=1e-4), \
        f"Kernel unexpectedly produced correct output in constant memory path with small num_features."

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_issue_non_contiguous_input():
    # Issue 2: The kernel does not check that the input tensor is contiguous.
    # Passing a non-contiguous tensor may lead to undefined data in the constant memory copy.
    # We intentionally pass a non-contiguous tensor and expect the result to differ from torch.softmax.
    batch_size = 4
    num_features = 512  # This will use constant memory since total elements = 2048 < MAX_CONST_INPUT_SIZE.
    x = torch.randn(batch_size, num_features, device="cuda", dtype=torch.float32)
    # Make x non-contiguous by transposing and then transposing back.
    x_noncontig = x.t().contiguous().t()
    # Force non-contiguity by taking a slice with a step.
    x_noncontig = x_noncontig[::, ::2]
    
    # Expand back to the original shape by repeating columns (this is artificial)
    # so we can at least call the kernel expecting a 2D tensor.
    x_faulty = x_noncontig.repeat(1,2)
    x_faulty = x_faulty.to(dtype=torch.float32)
    # Ensure the tensor is really non-contiguous.
    assert not x_faulty.is_contiguous(), "Test tensor unexpectedly became contiguous."
    
    module = build_kernel()
    y = module.forward(x_faulty)
    torch.cuda.synchronize()
    expected = torch.softmax(x_faulty, dim=1)
    # If the kernel mishandles non-contiguous memory, the result will be wrong.
    assert not torch.allclose(y, expected, atol=1e-4), \
        "Kernel unexpectedly handled non-contiguous input correctly (issue expected)."

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_issue_large_input_constant_memory_limit():
    # Issue 3: The constant memory kernel can process only when total elements <= MAX_CONST_INPUT_SIZE.
    # Here we provide an input that exceeds that limit so that the fallback global kernel is used.
    # However, if num_features is small (relative to THREADS_PER_BLOCK) the same uninitialized shared memory bug (issue 1)
    # may affect the global kernel as well.
    batch_size = 1024
    num_features = 20  # small number of features compared to THREADS_PER_BLOCK.
    # Total elements = 1024*20 = 20480 > MAX_CONST_INPUT_SIZE (16384), triggering global kernel.
    x = torch.randn(batch_size, num_features, device="cuda", dtype=torch.float32)
    module = build_kernel()
    y = module.forward(x)
    torch.cuda.synchronize()
    expected = torch.softmax(x, dim=1)
    # Due to the shared memory reduction bug (for small num_features), the result is expected to be incorrect.
    diff = (y - expected).abs().max().item()
    assert diff > 1e-3, f"Kernel global memory path produced output too close to expected (diff={diff}), bug not triggered."

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_issue_wrong_dtype():
    # Issue 4: The kernel requires float32 input. Passing a tensor of a different type (e.g. float64)
    # will trigger the TORCH_CHECK, leading to a RuntimeError.
    batch_size = 8
    num_features = 256
    x = torch.randn(batch_size, num_features, device="cuda", dtype=torch.float64)
    module = build_kernel()
    with pytest.raises(RuntimeError):
        # The forward function includes a TORCH_CHECK that ensures input is float32.
        y = module.forward(x)
