
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

# Issue 1: Number of elements not divisible by 4
def test_non_multiple_of_four():
    my_module = build_kernel()
    # Create tensors with a number of elements not divisible by 4.
    # For example, 130 elements (should be 128 or 132 for safe vectorized access).
    n = 130
    predictions = torch.randn(n, device="cuda", dtype=torch.float32)
    # For binary classification hinge loss, targets are expected to be -1 or 1.
    targets = (torch.randint(0, 2, (n,), device="cuda", dtype=torch.float32) * 2) - 1
    # The kernel uses reinterpret_cast<float4>, so it will process data in chunks of 4.
    # This test should trigger the unhandled tail elements.
    with pytest.raises(Exception):
        # We expect that the kernel may either crash or produce an error because of unprocessed tail.
        torch.mean(my_module.forward(predictions, targets)).item()

# Issue 2: Misaligned memory access due to non 16-byte alignment.
def test_memory_alignment_issue():
    my_module = build_kernel()
    # Create a tensor with extra element so that a slice starting from index 1 is misaligned.
    base = torch.randn(129, device="cuda", dtype=torch.float32)
    # Slicing from index 1 often produces a tensor that is still contiguous
    # but may have a data_ptr() that is not 16-byte aligned.
    predictions = base[1:]
    targets = ((torch.randint(0, 2, (predictions.numel(),), device="cuda", dtype=torch.float32) * 2) - 1)
    # The kernel does not account for misaligned addresses. This may lead to undefined behavior.
    with pytest.raises(Exception):
        torch.mean(my_module.forward(predictions, targets)).item()

# Issue 3: Input tensor dtype is not float32.
def test_dtype_issue():
    my_module = build_kernel()
    n = 128
    # Create tensors of dtype float64 instead of float32.
    predictions = torch.randn(n, device="cuda", dtype=torch.float64)
    targets = ((torch.randint(0, 2, (n,), device="cuda", dtype=torch.float64) * 2) - 1)
    # Although the CHECK_INPUT macro only checks device and contiguity,
    # using float64 will cause reinterpret_cast to float4 to misinterpret the data.
    with pytest.raises(Exception):
        torch.mean(my_module.forward(predictions, targets)).item()
        
if __name__ == "__main__":
    pytest.main([__file__])
