
import os
import pytest
import torch
from torch.utils.cpp_extension import load

# Build the CUDA kernel extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="kl_div_extension",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Helper function to compute correct KL divergence using PyTorch's F.kl_div.
# Note: torch.nn.functional.kl_div expects input to be log-probabilities.
def correct_kl_div(log_pred, target):
    # Computes: sum_i target_i * (log(target_i) - log_pred_i)
    # For reduction='batchmean' with a flat tensor, we sum and divide by n.
    # Here we simply compute elementwise and average.
    return (target * (target.log() - log_pred)).mean()

# Issue 1: Incorrect KL divergence formula.
# Create a test that compares the kernel’s output with the correct value.
# We expect the results to differ significantly.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_kl_div_formula_error():
    cuda_module = build_kernel()
    # small tensor so that reduction is simple and numerical error is visible
    batch_size = 32
    dim = 16
    # Use softmax to ensure probabilities
    predictions = torch.randn(batch_size, dim, device='cuda').softmax(dim=-1)
    targets = torch.randn(batch_size, dim, device='cuda').softmax(dim=-1)
    # Log predictions required by the kernel.
    log_predictions = predictions.log()
    # Run our kernel
    result_cuda = cuda_module.forward(log_predictions.contiguous(), targets.contiguous())
    # Compute correct forward using PyTorch's F.kl_div:
    result_ref = torch.nn.functional.kl_div(log_predictions, targets, reduction='batchmean')
    # The kernel formula is incorrect so they should differ. We assert they are not close.
    assert not torch.allclose(result_cuda, result_ref, atol=1e-4), (
        "Test failed: Kernel output matches the reference even though the formula is incorrect."
    )

# Issue 2: Inadequate reduction synchronization.
# Run the kernel with a size that forces many blocks and non-power-of-two thread counts.
# By repeatedly running the kernel, we check for inconsistent answers.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_reduction_sync_issue():
    cuda_module = build_kernel()
    # Choose a size that forces multiple blocks and non-even distribution of work.
    n = 1234567  # odd-sized total number of elements
    predictions = torch.randn(n, device='cuda').softmax(dim=0)
    targets = torch.randn(n, device='cuda').softmax(dim=0)
    log_predictions = predictions.log()
    # Run the kernel multiple times to see if the reduction gives consistent results.
    results = []
    for _ in range(10):
        res = cuda_module.forward(log_predictions.contiguous(), targets.contiguous())
        # Ensure synchronization explicitly.
        torch.cuda.synchronize()
        results.append(res.item())
    # If reduction synchronization is flawed, results might vary between runs.
    # Here we require a variation larger than a small epsilon.
    max_diff = max(results) - min(results)
    assert max_diff > 1e-3, (
        f"Test failed: Reduction seems to be correctly synchronized. Max diff: {max_diff:.6f}"
    )

# Issue 3: Ambiguous use of min in host code.
# Trigger the potential problem by ensuring that (n + threads - 1) / threads exceeds 1024.
# The kernel limits the block count to 1024; if the chosen size is huge, some elements may not be processed.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_min_usage_issue():
    cuda_module = build_kernel()
    # Create a huge tensor that in principle should use more than 1024 blocks if not clamped.
    n = 1 << 22  # ~4 million elements
    predictions = torch.randn(n, device='cuda').softmax(dim=0)
    targets = torch.randn(n, device='cuda').softmax(dim=0)
    log_predictions = predictions.log()
    result_cuda = cuda_module.forward(log_predictions.contiguous(), targets.contiguous())
    # Compute reference KL divergence
    result_ref = torch.nn.functional.kl_div(log_predictions, targets, reduction='batchmean')
    # With a proper block count, the results should match closely.
    # However, if the clamping via ambiguous min is wrong, the kernel result will be far off.
    diff = abs(result_cuda.item() - result_ref.item())
    assert diff > 1e-3, (
        f"Test failed: Expected a large error due to block count clamping issues from ambiguous min, got diff = {diff:.6f}"
    )

# Issue 4: Lack of kernel launch error checking.
# Pass in inputs with an unsupported data type (e.g. torch.float64) so that the kernel launch fails.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_kernel_launch_error_check():
    cuda_module = build_kernel()
    # Create double precision inputs.
    batch_size = 32
    dim = 16
    predictions = torch.randn(batch_size, dim, device='cuda', dtype=torch.float64).softmax(dim=-1)
    targets = torch.randn(batch_size, dim, device='cuda', dtype=torch.float64).softmax(dim=-1)
    log_predictions = predictions.log()
    # The kernel expects float pointers, so passing float64 should trigger an error.
    with pytest.raises(RuntimeError):
        _ = cuda_module.forward(log_predictions.contiguous(), targets.contiguous())
