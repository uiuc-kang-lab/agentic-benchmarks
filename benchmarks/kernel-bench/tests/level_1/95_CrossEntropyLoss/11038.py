
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Utility function to load (and rebuild if necessary) the CUDA extension
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Out‐of‐bound target index. Expect an error (or undefined behavior)
def test_out_of_bound_target():
    batch_size = 128
    num_classes = 10
    predictions = torch.randn(batch_size, num_classes, device="cuda", dtype=torch.float32)
    # Setting one target to an invalid value (e.g. num_classes) should be out‐of‐range.
    targets = torch.randint(0, num_classes, (batch_size,), device="cuda", dtype=torch.int64)
    # Intentionally assign an out-of-bound target for one sample.
    targets[0] = num_classes  # invalid index
    my_kernel = build_kernel()
    with pytest.raises(RuntimeError):
        # The undefined behavior should result in a CUDA error,
        # which is caught by TORCH_CHECK in the C++ code.
        my_kernel.forward(predictions, targets)
    torch.cuda.synchronize()

# Test 2: Non‐contiguous predictions tensor.
def test_non_contiguous_predictions():
    batch_size = 64
    num_classes = 10
    # Create a contiguous tensor and then create a non‐contiguous view by transposing and then transposing back.
    base = torch.randn(num_classes, batch_size, device="cuda", dtype=torch.float32)
    predictions = base.t()  # now predictions is (batch_size, num_classes) but non‐contiguous
    targets = torch.randint(0, num_classes, (batch_size,), device="cuda", dtype=torch.int64)
    
    my_kernel = build_kernel()
    # If the kernel does not properly handle non‐contiguous memory, the computed loss will differ from PyTorch's
    loss_kernel = my_kernel.forward(predictions, targets)
    loss_pt = torch.nn.functional.cross_entropy(predictions.contiguous(), targets)
    # They may differ due to memory layout issues.
    with pytest.raises(AssertionError):
        assert torch.allclose(loss_kernel, loss_pt, atol=1e-5), "Kernel output does not match PyTorch cross_entropy loss due to non‐contiguous input."
    torch.cuda.synchronize()

# Test 3: Reduction assumption violation: non–power‐of‐two blockDim.x.
# We simulate this by constructing a scenario where num_classes is very low relative to the fixed blockDim.x in the kernel.
def test_small_num_classes():
    # Because the kernel launch in the C++ host code fixes threads_x to 128,
    # setting num_classes very low (e.g. 2 or 1) will result in most threads doing no work.
    batch_size = 64
    num_classes = 2  # very small number of classes relative to threads_x=128
    predictions = torch.randn(batch_size, num_classes, device="cuda", dtype=torch.float32)
    targets = torch.randint(0, num_classes, (batch_size,), device="cuda", dtype=torch.int64)
    
    my_kernel = build_kernel()
    loss_kernel = my_kernel.forward(predictions, targets)
    loss_pt = torch.nn.functional.cross_entropy(predictions, targets)
    # Even if the kernel gives a valid result mathematically (due to fmaxf handling) it may be imprecise.
    # Here we check for a difference that signals potential reduction issues.
    assert not torch.allclose(loss_kernel, loss_pt, atol=1e-7), "Kernel reduction likely masks inefficiency or precision issues for very small num_classes."
    torch.cuda.synchronize()

# Test 4: Many idle threads in reduction (inefficiency/possible inaccuracies).
# We simulate this by setting num_classes far less than blockDim.x.
def test_num_classes_much_smaller_than_threads_x():
    # Use a scenario where num_classes is 3 while the kernel uses 128 threads in x-direction.
    batch_size = 128
    num_classes = 3
    predictions = torch.randn(batch_size, num_classes, device="cuda", dtype=torch.float32)
    targets = torch.randint(0, num_classes, (batch_size,), device="cuda", dtype=torch.int64)
    
    my_kernel = build_kernel()
    loss_kernel = my_kernel.forward(predictions, targets)
    loss_pt = torch.nn.functional.cross_entropy(predictions, targets)
    # Differences in the computed loss (if any) may indicate issues with handling many inactive threads.
    # We check that the results are not exactly equal (or that precision degrades) to trigger our warning.
    diff = (loss_kernel - loss_pt).abs().item()
    assert diff > 1e-7, "Kernel loss unexpectedly matches PyTorch loss exactly. Idle threads in reduction may be mishandled."
    torch.cuda.synchronize()
