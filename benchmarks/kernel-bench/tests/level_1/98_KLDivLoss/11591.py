
import torch
import pytest
from torch.nn import functional as F
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1: Incorrect KL divergence formula
def test_incorrect_formula():
    # Create a simple case where the error in the loss formula is evident.
    # Use a single-element tensor so that we can compute the expected value by hand.
    # Let log_pred = log(0.5) and target = 0.5.
    # Expected KL divergence according to PyTorch: loss = target*(log(target)-log_pred) = 0.5*(log(0.5) - log(0.5)) = 0.
    # The kernel, however, computes: exp(log_pred) - target*log_pred = 0.5 - 0.5*log(0.5), a non-zero value.
    device = torch.device("cuda")
    # Construct a tensor of one element. Note: use softmax to mimic the real scenario, though here we override.
    pred = torch.full((1, 1), 0.5, device=device, dtype=torch.float32)
    target = torch.full((1, 1), 0.5, device=device, dtype=torch.float32)
    # Compute reference KL divergence using PyTorch built-in (with reduction 'batchmean')
    log_pred = torch.log(pred)
    ref_loss = F.kl_div(log_pred, target, reduction='batchmean')
    
    module = build_kernel()
    # The kernel expects already computed log_predictions and targets.
    kernel_loss = module.forward(log_pred, target)
    torch.cuda.synchronize()
    # They should not be close since the kernel formula is wrong.
    assert not torch.allclose(kernel_loss, ref_loss, atol=1e-6), (
        f"Kernel loss seems to match the reference value, but an incorrect formula should produce a different result. "
        f"Kernel: {kernel_loss.item()}, Reference: {ref_loss.item()}"
    )

# Issue 2: Incorrect normalization factor (dividing by total number of elements instead of batch size)
def test_normalization_error():
    device = torch.device("cuda")
    # Create a batch of 4 samples with 8 classes each.
    batch_size = 4
    num_classes = 8
    # Create uniform distributions: predictions and targets as softmax
    # Set predictions such that log(prediction) is the same across examples.
    logits = torch.ones((batch_size, num_classes), device=device, dtype=torch.float32)
    pred = torch.softmax(logits, dim=-1)
    target = torch.softmax(torch.randn((batch_size, num_classes), device=device, dtype=torch.float32), dim=-1)
    
    log_pred = torch.log(pred)
    ref_loss = F.kl_div(log_pred, target, reduction='batchmean')  # divides by batch_size
    
    module = build_kernel()
    kernel_loss = module.forward(log_pred, target)
    torch.cuda.synchronize()
    # Because kernel divides by total number of elements (batch_size*num_classes),
    # its output will be lower than ref_loss by a factor of (num_classes).
    expected_kernel_loss = ref_loss / num_classes
    assert torch.allclose(kernel_loss, expected_kernel_loss, atol=1e-5), (
        f"Kernel normalization error: expected approx {expected_kernel_loss.item()}, but got {kernel_loss.item()}."
    )

# Issue 3a: Lack of check for non-contiguous inputs.
def test_non_contiguous_input():
    device = torch.device("cuda")
    batch_size = 16
    num_classes = 32
    # Create contiguous inputs first
    logits = torch.randn((batch_size, num_classes), device=device, dtype=torch.float32)
    pred = torch.softmax(logits, dim=-1)
    target = torch.softmax(torch.randn((batch_size, num_classes), device=device, dtype=torch.float32), dim=-1)
    log_pred = torch.log(pred)
    
    # Make a non-contiguous version by transposing (if possible) and then taking a view.
    log_pred_nc = log_pred.t().contiguous().t()  # This should be non-contiguous if we remove contiguous call...
    if log_pred_nc.is_contiguous():
        # Force non-contiguity by slicing stride.
        log_pred_nc = log_pred_nc[:, ::2].repeat(1, 2)
        target_nc = target[:, ::2].repeat(1, 2)
    else:
        target_nc = target

    module = build_kernel()
    try:
        kernel_loss_nc = module.forward(log_pred_nc, target_nc)
        torch.cuda.synchronize()
    except Exception as e:
        pytest.fail(f"Kernel failed on non-contiguous input: {e}")

# Issue 3b: Lack of check for input tensor type (only float32 is supported)
def test_wrong_dtype():
    device = torch.device("cuda")
    batch_size = 8
    num_classes = 16
    # Create double precision inputs instead of float32.
    logits = torch.randn((batch_size, num_classes), device=device, dtype=torch.float64)
    pred = torch.softmax(logits, dim=-1)
    target = torch.softmax(torch.randn((batch_size, num_classes), device=device, dtype=torch.float64), dim=-1)
    # Compute log and cast remains in double.
    log_pred = torch.log(pred)
    
    module = build_kernel()
    with pytest.raises(RuntimeError):
        # The kernel expects float32. Passing float64 should raise an error.
        _ = module.forward(log_pred, target)
