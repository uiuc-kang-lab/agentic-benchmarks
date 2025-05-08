
import pytest
import torch
from torch.utils.cpp_extension import load

# Build the CUDA extension
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test case 1:
# Passing input tensors with wrong dtype (e.g., float64 instead of float32)
def test_wrong_dtype():
    my_module = build_kernel()
    # Create a tensor of float64 for predictions and targets
    predictions = torch.randn(128, device="cuda", dtype=torch.float64)
    # Mimic binary classification targets of -1 or 1; float64 will be used here.
    targets = (torch.randint(0, 2, (128,), device="cuda", dtype=torch.int32).float() * 2 - 1).double()
    with pytest.raises(RuntimeError):
        # This should fail because the kernel expects float (float32) but got float64.
        out = my_module.forward(predictions, targets)
        # Synchronize to force any asynchronous error to be raised.
        torch.cuda.synchronize()

# Test case 2:
# Passing non-contiguous tensors, which should trigger the CHECK_CONTIGUOUS macros in the kernel.
def test_non_contiguous():
    my_module = build_kernel()
    predictions = torch.randn(128, 1, device="cuda", dtype=torch.float32)
    targets = (torch.randint(0, 2, (128,), device="cuda", dtype=torch.int32).float() * 2 - 1).to(dtype=torch.float32)
    # Make predictions non-contiguous via transposition or view manipulation.
    predictions_noncontig = predictions.t()  # transpose makes it non-contiguous in many cases.
    with pytest.raises(RuntimeError):
        out = my_module.forward(predictions_noncontig, targets)
        torch.cuda.synchronize()
