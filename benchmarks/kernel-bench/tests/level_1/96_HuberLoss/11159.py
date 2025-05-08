
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import numpy as np

# Function to (re)build the CUDA extension module from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Helper function: compute smooth L1 loss in Python
def smooth_l1_loss_ref(predictions, targets):
    diff = predictions - targets
    abs_diff = diff.abs()
    loss = torch.where(abs_diff < 1.0, 0.5 * diff * diff, abs_diff - 0.5)
    # Mean reduction over all elements
    return loss.mean()

# Issue 1 test: the kernel assumes float32.
# If we pass double tensors the kernel’s reinterpretation (via data_ptr<float>())
# will produce an incorrect result.
def test_input_tensor_type():
    # Create double-precision tensors.
    predictions = torch.randn(1024, dtype=torch.float64, device="cuda")
    targets = torch.randn(1024, dtype=torch.float64, device="cuda")
    module = build_kernel()
    # Call the CUDA kernel. It assumes float32 so the result will differ from the expected loss.
    device_loss = module.forward(predictions, targets)
    ref_loss = smooth_l1_loss_ref(predictions, targets)
    # The two results should not be equal due to misinterpretation of the memory.
    assert not torch.allclose(device_loss, ref_loss.float(), atol=1e-3), (
        f"Test failed: Kernel accepted double tensors, but results match! "
        f"Kernel output: {device_loss.item()}, reference loss: {ref_loss.item()}"
    )

# Issue 2 test: the kernel uses a fixed shared memory size and reduction over a fixed number 
# of warps. When the number of threads per block is not a multiple of the warp size,
# unused work-items (and corresponding shared memory locations) may not be properly zero‐initialized,
# potentially corrupting the final result.
#
# We trigger this by creating a case where the total number of elements is much smaller than 256 
# (the fixed block size) so that only an “incomplete” block is used.
def test_incomplete_thread_block():
    # Create a small input so that only one block is launched.
    # Note: Although the kernel launch always uses block_size=256, if n_elements is smaller,
    # then many threads in the block never work. This leaves some shared memory locations uninitialized.
    n_elements = 70  # Not a multiple of 32, so the final warp in the block is incomplete.
    predictions = torch.randn(n_elements, dtype=torch.float32, device="cuda")
    targets = torch.randn(n_elements, dtype=torch.float32, device="cuda")
    module = build_kernel()
    device_loss = module.forward(predictions, targets)
    ref_loss = smooth_l1_loss_ref(predictions, targets)
    # Because of the reduction issue, the kernel result is expected to differ from the reference.
    assert not torch.allclose(device_loss, ref_loss, atol=1e-5), (
        f"Test failed: Kernel reduction error not triggered. "
        f"Kernel output: {device_loss.item()}, reference loss: {ref_loss.item()}"
    )
