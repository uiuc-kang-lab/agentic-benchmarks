
import pytest
import torch
from torch.utils.cpp_extension import load

# Function to build the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# A helper to call the forward function from the extension and enforce synchronization.
def run_forward(kernel_module, predictions, targets):
    result = kernel_module.forward(predictions, targets)
    torch.cuda.synchronize()  # trigger any delayed CUDA errors
    return result

# Test case 1: Passing tensors with incorrect type (e.g., float64) to trigger issue #1.
def test_wrong_dtype():
    kernel_module = build_kernel()
    batch_size = 128
    # Create inputs as float64 instead of float32.
    predictions = torch.randn(batch_size, 1, dtype=torch.float64, device="cuda")
    # Targets as float64
    targets = (torch.randint(0, 2, (batch_size,), device="cuda", dtype=torch.float64) * 2 - 1)
    with pytest.raises(Exception):
        # This should raise an error or produce undefined behavior.
        run_forward(kernel_module, predictions, targets)

# Test case 2: Mismatched tensor sizes to trigger issue #2.
def test_mismatched_shapes():
    kernel_module = build_kernel()
    # Create predictions with one extra element.
    predictions = torch.randn(129, 1, dtype=torch.float32, device="cuda")
    # Create targets with fewer elements.
    targets = (torch.randint(0, 2, (128,), device="cuda", dtype=torch.float32) * 2 - 1)
    # The kernel does not check shape consistency so it may silently corrupt memory.
    # We force an error by expecting an out-of-bound access error or a CUDA error.
    with pytest.raises(Exception):
        run_forward(kernel_module, predictions, targets)

# Test case 3: Passing non-contiguous tensors to trigger issue #3.
def test_non_contiguous_inputs():
    kernel_module = build_kernel()
    batch_size = 128
    # Create a contiguous tensor and then slice it to make it non-contiguous.
    predictions_full = torch.randn(batch_size * 2, 1, dtype=torch.float32, device="cuda")
    targets_full = (torch.randint(0, 2, (batch_size * 2,), device="cuda", dtype=torch.float32) * 2 - 1)
    # Make non-contiguous by taking every other element.
    predictions = predictions_full[::2]
    targets = targets_full[::2]
    # The CHECK_INPUT macros should catch non-contiguous tensors.
    with pytest.raises(Exception):
        run_forward(kernel_module, predictions, targets)

# Test case 4: Kernel launch error checking is missing, so we attempt to force an error by
# using an extremely large tensor size (if allowed by device memory, this may cause a failure).
def test_kernel_launch_error():
    kernel_module = build_kernel()
    # Try using an input tensor that is too large. 
    # This might not always trigger an error, but on most systems it should.
    try:
        # Let's create a tensor with a huge number of elements.
        n = 10**8  # Adjust this number if necessary to trigger an error
        predictions = torch.randn(n, dtype=torch.float32, device="cuda")
        targets = (torch.randint(0, 2, (n,), device="cuda", dtype=torch.float32) * 2 - 1)
        # We expect the kernel to fail due to launch configuration or memory issues.
        with pytest.raises(Exception):
            run_forward(kernel_module, predictions, targets)
    except RuntimeError:
        # If a RuntimeError is raised during tensor allocation, then this test has served its purpose.
        pass

if __name__ == "__main__":
    pytest.main([__file__])
