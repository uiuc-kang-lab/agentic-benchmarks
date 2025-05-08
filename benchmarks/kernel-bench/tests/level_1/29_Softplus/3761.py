
import pytest
import torch
from torch.utils.cpp_extension import load

# Function to build and load the CUDA extension from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="softplus_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test case for Issue 1: Half precision is not supported.
def test_half_precision_support():
    kernel_mod = build_kernel()
    # Create a half precision input tensor
    x = torch.randn(16, 16384, device="cuda", dtype=torch.half)
    with pytest.raises(RuntimeError):
        # Expect the dispatch (or the kernel launch) to fail because half
        # precision is not handled by AT_DISPATCH_FLOATING_TYPES in our kernel.
        _ = kernel_mod.forward(x)
        
# Test case for Issue 2: Non-contiguous tensors are not handled correctly.
def test_non_contiguous_tensor():
    kernel_mod = build_kernel()
    # Create a contiguous tensor and then make it non-contiguous via a transpose
    x = torch.randn(16, 16384, device="cuda", dtype=torch.float32)
    # Create a non contiguous tensor by transposing a 2D view; for example, use shape (16384, 16)
    x_noncontig = x.t()
    # Compute expected softplus using PyTorch's in-built function.
    expected = torch.nn.functional.softplus(x_noncontig)
    # The kernel uses x.data_ptr() assuming a contiguous layout.
    # So it may produce different/wrong results.
    result = kernel_mod.forward(x_noncontig)
    torch.cuda.synchronize()
    # Instead of asserting equality, we check that the outputs do not match.
    # (If the kernel assumed a contiguous layout, non-contiguous input leads to errors.)
    assert not torch.allclose(result, expected, atol=1e-5), (
        "Kernel produced matching results on a non-contiguous tensor "
        "which is unexpected."
    )

# Test case for Issue 3: Missing error checking after kernel launch.
def test_kernel_launch_error_checking():
    kernel_mod = build_kernel()
    # We simulate an error condition by passing a tensor with no elements.
    # Although an empty tensor might be handled gracefully, it is used here to
    # show that the kernel does not perform additional error checking.
    x_empty = torch.empty(0, device="cuda", dtype=torch.float32)
    result = kernel_mod.forward(x_empty)
    # Force a CUDA synchronization which would propagate any async kernel launch errors.
    try:
        torch.cuda.synchronize()
    except Exception as e:
        pytest.fail(f"Kernel launch error was not caught: {e}")
    # If no error was caught, we assume the absence of explicit error checking.
    # (The test passes if no CUDA error is raised.)
    assert result.numel() == 0, "Empty input should result in empty output."

if __name__ == "__main__":
    pytest.main([__file__])
