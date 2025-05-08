
import torch
import pytest
from torch.utils.cpp_extension import load
import os

# Build the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# A helper function that computes PyTorch reference KL divergence 
def reference_kl_div(log_predictions, targets):
    # PyTorch's kl_div expects log_predictions and targets.
    # Note: The built-in function does not include the target*log(target) constant term.
    # Here we use the same torch.nn.functional.kl_div for reduction 'batchmean'
    return torch.nn.functional.kl_div(log_predictions, targets, reduction='batchmean')

# Issue 1: Wrong tensor data type (non-float32) or mis‐aligned memory.
def test_input_dtype_alignment():
    my_module = build_kernel()
    N = 1024
    # Create double precision inputs. The kernel expects float32.
    predictions = torch.randn(N, device="cuda", dtype=torch.float64).softmax(dim=-1)
    targets = torch.randn(N, device="cuda", dtype=torch.float64).softmax(dim=-1)
    # Convert to log probabilities for consistency with the host code.
    log_preds = torch.log(predictions)
    with pytest.raises(RuntimeError):
        # This should trigger an error because the kernel uses float pointer arithmetic.
        my_module.forward(log_preds, targets)

# Issue 2: Non-contiguous tensor input.
def test_non_contiguous_inputs():
    my_module = build_kernel()
    batch_size, input_size = 128, 4096
    predictions = torch.randn(batch_size, input_size, device="cuda", dtype=torch.float32).softmax(dim=-1)
    targets = torch.randn(batch_size, input_size, device="cuda", dtype=torch.float32).softmax(dim=-1)
    # Make non-contiguous tensors by transposing or slicing
    predictions_nc = predictions.t().clone()  # t() makes it non-contiguous; then clone() to keep same memory layout (but non-contiguous)
    targets_nc = targets.t().clone()
    # Take log after making non-contiguous.
    log_preds_nc = torch.log(predictions_nc)
    with pytest.raises(RuntimeError):
        my_module.forward(log_preds_nc, targets_nc)

# Issue 3: Incorrect mathematical formulation.
def test_incorrect_math():
    my_module = build_kernel()
    # Use a small tensor where the math can be computed.
    batch_size, input_size = 8, 16
    predictions = torch.full((batch_size, input_size), 0.125, device="cuda", dtype=torch.float32)
    targets = torch.full((batch_size, input_size), 0.125, device="cuda", dtype=torch.float32)
    # Ensure inputs sum to 1 on the last dimension
    predictions = predictions / predictions.sum(dim=-1, keepdim=True)
    targets = targets / targets.sum(dim=-1, keepdim=True)
    # Compute host reference result using PyTorch's kl_div
    log_preds = torch.log(predictions)
    ref = reference_kl_div(log_preds, targets)
    # Call our kernel (which implements exp(lp) - tt*lp)
    kernel_out = my_module.forward(log_preds, targets)
    # They should not be equal because the kernel's math is different.
    assert not torch.allclose(kernel_out, ref, atol=1e-5), \
        f"Kernel math unexpectedly matches reference. Kernel output: {kernel_out}, Reference: {ref}"

# Issue 4: Incorrect normalization in reduction.
def test_incorrect_normalization():
    my_module = build_kernel()
    # Create a case with distinct batch size and total element count.
    batch_size, input_size = 4, 100  # total elements = 400
    # Use a tensor with non-trivial values.
    predictions = torch.randn(batch_size, input_size, device="cuda", dtype=torch.float32).softmax(dim=-1)
    targets = torch.randn(batch_size, input_size, device="cuda", dtype=torch.float32).softmax(dim=-1)
    log_preds = torch.log(predictions)
    ref = reference_kl_div(log_preds, targets)
    kernel_out = my_module.forward(log_preds, targets)
    # Because the kernel divides the accumulated sum by total elements (400) instead of batch size (4),
    # the scaling factor is off by 100. Therefore, the kernel output is ref/100.
    # We test that the kernel output is not close to the expected batchmean reduction.
    expected_scaled = ref  # expected from torch.nn.functional.kl_div
    incorrect_scaled = kernel_out
    # We expect a significant relative difference.
    rel_diff = torch.abs(incorrect_scaled - expected_scaled) / torch.abs(expected_scaled + 1e-6)
    assert rel_diff > 0.5, \
        f"Kernel normalization may be correct unexpectedly. Kernel output: {incorrect_scaled}, Reference: {expected_scaled}"

if __name__ == "__main__":
    pytest.main([__file__])
