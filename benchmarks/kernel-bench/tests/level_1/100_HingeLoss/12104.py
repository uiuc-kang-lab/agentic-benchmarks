
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="hinge_loss_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Helper: reference hinge loss using PyTorch (without kernel)
def hinge_loss_ref(predictions, targets):
    return torch.mean(torch.clamp(1 - predictions * targets, min=0))

# Issue 1: Mismatched number of elements (broadcast incompatibility)
def test_mismatched_shapes():
    # predictions has 128 elements, targets has 64 elements.
    predictions = torch.randn(128, device="cuda", dtype=torch.float32)
    targets = torch.randint(0, 2, (64,), device="cuda", dtype=torch.float32) * 2 - 1
    kernel_module = build_kernel()
    # Since the kernel uses predictions.numel() (128) but targets only has 64 elements,
    # it will access out‐of‐bounds memory.
    with pytest.raises(RuntimeError):
        # This should trigger an illegal memory access / kernel error.
        loss = kernel_module.forward(predictions, targets)
        torch.cuda.synchronize()

# Issue 2: Input data type is not float32
def test_incorrect_dtype():
    # Use double precision tensors even though the kernel expects float32.
    predictions = torch.randn(128, device="cuda", dtype=torch.float64)
    targets = (torch.randint(0, 2, (128,), device="cuda", dtype=torch.float64) * 2 - 1)
    kernel_module = build_kernel()
    # The kernel will reinterpret the double memory as float. The result will
    # not match the reference computed in double precision.
    loss_kernel = kernel_module.forward(predictions, targets)
    loss_ref = hinge_loss_ref(predictions.float(), targets.float())
    # The values will likely be substantially different.
    assert not torch.allclose(loss_kernel, loss_ref, atol=1e-3), "Kernel accepted non-float32 input unexpectedly"

# Issue 3: Empty input tensor (n == 0)
def test_empty_input():
    predictions = torch.empty(0, device="cuda", dtype=torch.float32)
    targets = torch.empty(0, device="cuda", dtype=torch.float32)
    kernel_module = build_kernel()
    # If n==0, the kernel reduction will use uninitialized shared memory,
    # leading to undefined behavior. We expect either an error or a nonsensical result.
    with pytest.raises(RuntimeError):
        loss = kernel_module.forward(predictions, targets)
        torch.cuda.synchronize()

# Issue 4: Non contiguous inputs
def test_non_contiguous_input():
    # Create contiguous float32 inputs.
    predictions = torch.randn(128, 1, device="cuda", dtype=torch.float32)
    targets = (torch.randint(0, 2, (128,), device="cuda", dtype=torch.float32) * 2 - 1).unsqueeze(1)
    # Make one of the tensors non-contiguous by transposing a 2D slice.
    predictions_noncontig = predictions.t()
    # The CHECK_CONTIGUOUS macro in the kernel is expected to trigger an error.
    kernel_module = build_kernel()
    with pytest.raises(RuntimeError):
        loss = kernel_module.forward(predictions_noncontig, targets)
        torch.cuda.synchronize()
