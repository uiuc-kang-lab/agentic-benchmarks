
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import math

def build_kernel():
    cuda_module = load(
        name="kl_div_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Helper function to compute the correct KL divergence using PyTorch definition.
# Given that F.kl_div expects log_predictions and targets,
# the “correct” per-element loss is: target * (log(target) - log_pred)
# and reduction 'batchmean' divides by batch_size.
def kl_div_torch(log_predictions, targets, batch_size):
    loss = F.kl_div(log_predictions, targets, reduction='batchmean')
    # Note: batchmean reduction sums over all samples and divides by batch_size.
    return loss

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_incorrect_formula():
    """
    Test that the kernel's computed output does not match the expected KL divergence
    due to the wrong element‐wise formula.
    """
    my_module = build_kernel()

    # Create a synthetic case with non-uniform non-zero tensors.
    batch_size, dim = 8, 16
    pred = torch.rand(batch_size, dim, device="cuda", dtype=torch.float32)
    pred = pred / pred.sum(dim=-1, keepdim=True)  # softmax-like normalization
    targets = torch.rand(batch_size, dim, device="cuda", dtype=torch.float32)
    targets = targets / targets.sum(dim=-1, keepdim=True)

    log_pred = torch.log(pred)
    # Launch kernel via the extension; note kernel returns a tensor of shape [1] 
    # that (incorrectly) divides by total n elements.
    kernel_out = my_module.forward(log_pred, targets)
    
    # Compute reference using PyTorch's built-in function.
    ref_loss = kl_div_torch(log_pred, targets, batch_size=batch_size)
    
    # Because of the wrong formula, the kernel output should differ significantly.
    assert not torch.allclose(kernel_out, ref_loss, atol=1e-3), (
        f"Kernel output unexpectedly matches the reference. Kernel: {kernel_out.item()}, Ref: {ref_loss.item()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_incorrect_normalization():
    """
    Test that the normalization in the kernel is inconsistent with 'batchmean'.
    For a toy example where the batch size and element count differ, the two outputs differ.
    """
    my_module = build_kernel()

    # Create inputs where each sample has multiple elements.
    batch_size, dim = 4, 1024  # total elements = 4096 per batch
    pred = torch.rand(batch_size, dim, device="cuda", dtype=torch.float32)
    pred = pred / pred.sum(dim=-1, keepdim=True)
    targets = torch.rand(batch_size, dim, device="cuda", dtype=torch.float32)
    targets = targets / targets.sum(dim=-1, keepdim=True)

    log_pred = torch.log(pred)
    kernel_out = my_module.forward(log_pred, targets)
    
    # PyTorch KLDivLoss with batchmean divides by batch size.
    ref_loss = kl_div_torch(log_pred, targets, batch_size=batch_size)
    
    # The kernel divides by total number of elements; hence for dim > 1, the loss scales wrongly.
    # Verify that the kernel output is off by an expected factor: nearly 1/(dim) times the correct loss.
    expected_factor = 1.0 / dim  # since kernel divides by (batch_size*dim) vs batch_size.
    
    diff = abs(kernel_out.item() - ref_loss.item())
    # We expect a notable difference due to the normalization mismatch.
    assert diff > 1e-3, (
        f"Normalization error not detected: Kernel: {kernel_out.item()}, Ref: {ref_loss.item()}, diff: {diff}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_unsupported_dtype():
    """
    Test that passing double precision tensors triggers a wrong output.
    The kernel internally casts pointers to float, so double precision inputs will be misinterpreted.
    """
    my_module = build_kernel()

    batch_size, dim = 8, 16
    pred = torch.rand(batch_size, dim, device="cuda", dtype=torch.float64)
    pred = pred / pred.sum(dim=-1, keepdim=True)
    targets = torch.rand(batch_size, dim, device="cuda", dtype=torch.float64)
    targets = targets / targets.sum(dim=-1, keepdim=True)

    log_pred = torch.log(pred)
    # The kernel expects float32. Casting to float32 would hide the bug.
    # Instead, we deliberately do not convert them and see that the result is not close.
    kernel_out = my_module.forward(log_pred, targets)
    
    ref_loss = F.kl_div(log_pred.float(), targets.float(), reduction='batchmean')
    
    # Since the kernel misinterprets the double memory, the outputs should differ significantly.
    assert not torch.allclose(kernel_out, ref_loss, atol=1e-3), (
        f"Kernel unexpectedly handled double inputs: Kernel: {kernel_out.item()}, Ref: {ref_loss.item()}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_non_contiguous_input():
    """
    Test that non‐contiguous inputs (resulting for example from a transpose)
    lead to incorrect behavior since the kernel assumes contiguous memory.
    """
    my_module = build_kernel()

    batch_size, dim = 8, 16
    # Create contiguous inputs first.
    pred = torch.rand(batch_size, dim, device="cuda", dtype=torch.float32)
    pred = pred / pred.sum(dim=-1, keepdim=True)
    targets = torch.rand(batch_size, dim, device="cuda", dtype=torch.float32)
    targets = targets / targets.sum(dim=-1, keepdim=True)
    
    # Make the tensors non-contiguous by transposing (which makes them 2-d tensor with swapped strides)
    pred_nc = pred.t()
    targets_nc = targets.t()
    # Now take log; note, these are still non-contiguous.
    log_pred_nc = torch.log(pred_nc)
    
    kernel_out = my_module.forward(log_pred_nc, targets_nc)
    
    # Compute reference on contiguous tensors.
    ref_loss = F.kl_div(torch.log(pred), targets, reduction='batchmean')
    
    assert not torch.allclose(kernel_out, ref_loss, atol=1e-3), (
        f"Kernel unexpectedly handled non-contiguous memory correctly: Kernel: {kernel_out.item()}, Ref: {ref_loss.item()}"
    )
