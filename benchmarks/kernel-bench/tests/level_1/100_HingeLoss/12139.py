
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="hinge_loss_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1: Mismatched sizes for predictions and targets.
def test_mismatched_shape():
    my_module = build_kernel()
    # Create predictions with 128 elements and targets with 64 elements (mismatch)
    predictions = torch.randn(128, device="cuda", dtype=torch.float32)
    targets = torch.randn(64, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        # This will likely trigger an invalid memory access (or wrong results)
        out = my_module.forward(predictions, targets)
        torch.cuda.synchronize()

# Issue 2: Unsupported tensor data type.
def test_incorrect_dtype():
    my_module = build_kernel()
    # Create tensors of type float64 instead of float32.
    predictions = torch.randn(128, device="cuda", dtype=torch.float64)
    targets = torch.randn(128, device="cuda", dtype=torch.float64)
    with pytest.raises(RuntimeError):
        out = my_module.forward(predictions, targets)
        torch.cuda.synchronize()

# Issue 3: Lack of kernel launch error checking.
# We simulate an error scenario by feeding non-contiguous tensors.
# The kernel enforces a contiguity check via CHECK_CONTIGUOUS, so we trigger that.
def test_non_contiguous_input():
    my_module = build_kernel()
    # Create a contiguous tensor and then create a non-contiguous version by transposing.
    predictions_base = torch.randn(16, 8, device="cuda", dtype=torch.float32)
    targets_base = torch.randint(0, 2, (16, 8), device="cuda").float() * 2 - 1
    predictions = predictions_base.t()  # Not contiguous
    targets = targets_base.t()            # Not contiguous
    with pytest.raises(RuntimeError):
        out = my_module.forward(predictions, targets)
        torch.cuda.synchronize()
