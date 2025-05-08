
import torch
import pytest
from torch.utils.cpp_extension import load
import math

def build_kernel():
    cuda_module = load(
        name="kl_div_kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Helper reference implementation using PyTorch's kl_div for log-probabilities,
# so we can compare against the expected value.
def kl_div_reference(log_predictions, targets):
    # PyTorch expects: loss = sum(target * (log(target) - log_predictions)) / batch_size
    # Here we mimic reduction='batchmean', i.e. dividing by batch size (first dimension)
    batch_size = log_predictions.size(0)
    # Avoid issues with zero values.
    safe_targets = torch.clamp(targets, min=1e-8)
    loss = (safe_targets * (torch.log(safe_targets) - log_predictions)).sum() / float(batch_size)
    return loss

@pytest.fixture(scope="module")
def kernel_module():
    return build_kernel()

# Issue 1:
# Test that passing non-float32 tensors (e.g., float64) causes error or incorrect results.
def test_input_tensor_type(kernel_module):
    batch_size = 128
    input_shape = (4096,)
    # Create float64 tensors.
    pred = torch.randn(batch_size, *input_shape, dtype=torch.float64, device="cuda").softmax(dim=-1)
    target = torch.randn(batch_size, *input_shape, dtype=torch.float64, device="cuda").softmax(dim=-1)
    # The kernel expects float32.
    with pytest.raises(RuntimeError):
        # If the extension does not check the type, the use of data_ptr<float>() will interpret
        # the 8-byte floats as 4-byte floats, leading to errors.
        _ = kernel_module.forward(pred, target)
        
# Issue 2:
# Test that non-contiguous or mis-aligned memory causes the kernel to produce an incorrect result.
def test_misaligned_memory(kernel_module):
    batch_size = 16
    input_shape = (103)  # Use a size that is not a multiple of 4 to stress tail processing and alignment.
    pred = torch.randn(batch_size, input_shape, device="cuda", dtype=torch.float32).softmax(dim=-1)
    target = torch.randn(batch_size, input_shape, device="cuda", dtype=torch.float32).softmax(dim=-1)
    
    # Create non-contiguous tensors by taking a narrow slice.
    pred_noncontig = pred[:, 1:]
    target_noncontig = target[:, 1:]
    
    # Call our kernel with misaligned inputs.
    result = kernel_module.forward(pred_noncontig.log(), target_noncontig)
    # Compute reference using PyTorch's built-in functional (note: the kernel arithmetic is wrong,
    # so we know it will differ from the correct value, but here we test that non-contiguous tensors lead
    # to further deviation from the reference).
    ref = torch.nn.functional.kl_div(pred_noncontig.log(), target_noncontig, reduction="batchmean")
    
    # Since the kernel does not check alignment, the result might be different.
    # We assert that the values are not close, indicating the misaligned access issue.
    assert not torch.allclose(result, ref, atol=1e-4), "Kernel produced a result similar to reference even with misaligned inputs."

# Issue 3 and 4:
# Test that the kernel computes a result that deviates from PyTorch's kl_div operator because of
# both the incorrect formula (missing target*log(target)) and the wrong normalization factor.
def test_incorrect_kl_divergence_calculation(kernel_module):
    batch_size = 128
    input_shape = (4096,)
    # Generate distributions.
    predictions = torch.randn(batch_size, *input_shape, device="cuda", dtype=torch.float32).softmax(dim=-1)
    targets = torch.randn(batch_size, *input_shape, device="cuda", dtype=torch.float32).softmax(dim=-1)
    
    # Prepare log predictions for kernel input.
    log_predictions = predictions.log()
    
    # Get result from kernel.
    kernel_result = kernel_module.forward(log_predictions, targets)
    
    # Reference KL divergence computed using PyTorch's functional kl_div.
    ref_result = torch.nn.functional.kl_div(log_predictions, targets, reduction="batchmean")
    
    # The kernel result is computed as:
    # sum(exp(log_predictions) - targets*log_predictions) / n
    # which is different from reference: sum(target*(log(target)-log_predictions)) / batch_size.
    # Therefore, we expect a significant deviation.
    diff = torch.abs(kernel_result - ref_result).item()
    
    # We threshold the difference. It is problem dependent but if the kernel were correct, the values
    # should agree closely. Here, expect a difference larger than a small tolerance.
    assert diff > 1e-3, f"Kernel appears to compute KL divergence correctly (diff={diff}) whereas it should be wrong."

