
import torch
import pytest
from torch.nn import functional as F
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="kl_cuda_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

def test_incorrect_kl_formula():
    """
    Test case to trigger the issue with the incorrect KL divergence formula.
    The expected KL divergence (computed by PyTorch's F.kl_div using log_predictions and target)
    is based on target*(log(target)-log_predictions). The kernel computes
    predictions - target*log_predictions (using expf to recover predictions) and misses
    the target*log(target) term.
    """
    kernel_mod = build_kernel()
    # Create a realistic batch of probability distributions.
    batch_size = 128
    input_shape = (4096,)
    pred = torch.randn(batch_size, *input_shape, device='cuda').softmax(dim=-1)
    target = torch.randn(batch_size, *input_shape, device='cuda').softmax(dim=-1)
    log_pred = torch.log(pred)
    
    # Flatten the input for the kernel (which expects a 1D array)
    log_pred_flat = log_pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    
    # Run the custom kernel forward pass.
    out_kernel = kernel_mod.forward(log_pred_flat, target_flat)
    
    # Compute the reference loss using PyTorch (kl_div with reduction 'batchmean')
    loss_ref = F.kl_div(log_pred, target, reduction='batchmean')
    
    # The outputs should not match due to the missing target*log(target) term.
    assert not torch.allclose(out_kernel, loss_ref, atol=1e-4), \
           f"Kernel unexpectedly produced the correct KL divergence! Kernel output: {out_kernel.item()}, ref: {loss_ref.item()}."

def test_wrong_reduction_scaling():
    """
    Test case to trigger the issue with the wrong reduction scaling.
    The kernel divides the accumulated loss by the total number of elements (n)
    while PyTorch's kl_div (with reduction 'batchmean') divides by the batch size.
    With small tensors (so n != batch_size) the discrepancy will be noticeable.
    """
    kernel_mod = build_kernel()
    # Use a smaller input shape so that the total number of elements is different from the batch size.
    batch_size = 32
    input_shape = (64,)
    pred = torch.randn(batch_size, *input_shape, device='cuda').softmax(dim=-1)
    target = torch.randn(batch_size, *input_shape, device='cuda').softmax(dim=-1)
    log_pred = torch.log(pred)
    
    # Flatten the input for the kernel.
    log_pred_flat = log_pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    
    # Run the custom kernel forward pass.
    out_kernel = kernel_mod.forward(log_pred_flat, target_flat)
    
    # Compute the reference loss using PyTorch (kl_div with 'batchmean' reduction)
    loss_ref = F.kl_div(log_pred, target, reduction='batchmean')
    
    # Because the kernel divides by n (total number of elements) rather than by the batch size,
    # its output should differ from the PyTorch result.
    assert not torch.allclose(out_kernel, loss_ref, atol=1e-4), \
           f"Kernel unexpectedly produced the correct reduction scaling! Kernel output: {out_kernel.item()}, ref: {loss_ref.item()}."
