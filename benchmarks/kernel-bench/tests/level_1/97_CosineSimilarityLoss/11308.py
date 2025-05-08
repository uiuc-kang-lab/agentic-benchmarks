
import torch
import pytest
from torch.utils.cpp_extension import load

# Build the CUDA kernel module from kernel.cu
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test case 1: Pass non-float32 input tensors to trigger type incompatibility
def test_non_float32_input():
    kernel = build_kernel()
    predictions = torch.randn(128, 4096, dtype=torch.float64, device="cuda")
    targets = torch.randn(128, 4096, dtype=torch.float64, device="cuda")
    with pytest.raises(RuntimeError):
        # Should trigger a TORCH_CHECK failure due to wrong dtype.
        kernel.forward(predictions, targets)

# Test case 2: Pass tensors with incorrect dimensions (non-2D) to trigger dimension check failure
def test_incorrect_dimensions():
    kernel = build_kernel()
    # 1D tensors are not allowed – the kernel expects 2D tensors.
    predictions = torch.randn(4096, dtype=torch.float32, device="cuda")
    targets = torch.randn(4096, dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError):
        kernel.forward(predictions, targets)

# Test case 3: Pass non-contiguous tensors to check that the kernel fails or produces incorrect results.
# The kernel’s pointer arithmetic assumes contiguity.
def test_non_contiguous_inputs():
    kernel = build_kernel()
    # Create contiguous tensors first.
    predictions = torch.randn(128, 4096, dtype=torch.float32, device="cuda")
    targets = torch.randn(128, 4096, dtype=torch.float32, device="cuda")
    # Make them non-contiguous by transposing twice in a way that breaks contiguity.
    predictions = predictions.t().clone().t()
    targets = targets.t().clone().t()
    # Although the shapes remain 2D, the memory layout is not as expected.
    loss_kernel = kernel.forward(predictions, targets)
    # Compute reference loss using PyTorch’s own implementation.
    loss_ref = torch.mean(1 - torch.nn.functional.cosine_similarity(predictions, targets, dim=1))
    # The outputs are expected to differ due to assumptions on contiguity.
    assert not torch.allclose(loss_kernel, loss_ref, atol=1e-5), (
        "Kernel loss should differ when inputs are non-contiguous."  
    )

# Test case 4: Test a case with D much smaller than the block size assumption.
# For D=16, the host code dispatches BLOCK_SIZE=32 which may be inefficient or behave unexpectedly.
def test_small_feature_dimension():
    kernel = build_kernel()
    predictions = torch.randn(128, 16, dtype=torch.float32, device="cuda")
    targets = torch.randn(128, 16, dtype=torch.float32, device="cuda")
    loss_kernel = kernel.forward(predictions, targets)
    loss_ref = torch.mean(1 - torch.nn.functional.cosine_similarity(predictions, targets, dim=1))
    # For this test we expect a mismatch if the kernel’s tuning logic isn’t well adapted.
    assert not torch.allclose(loss_kernel, loss_ref, atol=1e-5), (
        "Kernel loss should be affected by inefficient thread utilization for small D."
    )

# Test case 5: Pass empty input tensors (N==0) to trigger potential undefined behavior.
def test_empty_inputs():
    kernel = build_kernel()
    predictions = torch.empty((0, 4096), dtype=torch.float32, device="cuda")
    targets = torch.empty((0, 4096), dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError):
        # Expect failure or an error due to degenerate (empty) inputs.
        kernel.forward(predictions, targets)
