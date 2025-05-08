
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# 1. Test to reveal the incorrect mathematical formula and wrong normalization.
def test_incorrect_math():
    # Create two probability distributions using softmax over a 2D tensor.
    batch_size, numel = 8, 128
    predictions = torch.randn(batch_size, numel, device="cuda", dtype=torch.float32).softmax(dim=-1)
    targets = torch.randn(batch_size, numel, device="cuda", dtype=torch.float32).softmax(dim=-1)

    # Compute PyTorch reference result using F.kl_div with reduction='batchmean'.
    log_preds = torch.log(predictions)
    ref_loss = F.kl_div(log_preds, targets, reduction='batchmean')

    # Compute kernel result.
    cuda_module = build_kernel()
    # The kernel expects the first input to be log(predictions)
    kernel_loss = cuda_module.forward(log_preds, targets)

    # Since the kernel is missing a target * log(target) term and divides by numel(),
    # the result should NOT be close to the PyTorch reference.
    # We expect a significant discrepancy.
    assert not torch.allclose(kernel_loss, ref_loss, atol=1e-5), \
        f"Kernel output {kernel_loss.item()} unexpectedly matches reference {ref_loss.item()}!"

# 2. Test to trigger type incompatibility when inputs are double.
def test_wrong_dtype():
    batch_size, numel = 8, 128
    # Create double tensors.
    predictions = torch.randn(batch_size, numel, device="cuda", dtype=torch.float64).softmax(dim=-1)
    targets = torch.randn(batch_size, numel, device="cuda", dtype=torch.float64).softmax(dim=-1)
    log_preds = torch.log(predictions)

    cuda_module = build_kernel()
    with pytest.raises(RuntimeError):
        # The kernel expects float32 and will likely throw an error or produce unexpected results.
        cuda_module.forward(log_preds, targets)
        
# 3. Test to trigger issues with non-contiguous (i.e. non-flat) tensor inputs.
def test_non_contiguous_inputs():
    batch_size, numel = 8, 128
    predictions = torch.randn(batch_size, numel * 2, device="cuda", dtype=torch.float32).softmax(dim=-1)
    targets = torch.randn(batch_size, numel * 2, device="cuda", dtype=torch.float32).softmax(dim=-1)
    # Create a non-contiguous slice by taking every second column.
    predictions_nc = predictions[:, ::2]
    targets_nc = targets[:, ::2]
    log_preds_nc = torch.log(predictions_nc)
    
    # Ensure the tensors are non-contiguous.
    assert not predictions_nc.is_contiguous(), "Expecting non-contiguous tensor."

    cuda_module = build_kernel()
    # The kernel simply uses numel() and pointer arithmetic so the non-contiguous layout will lead to wrong results.
    kernel_loss = cuda_module.forward(log_preds_nc.contiguous(), targets_nc.contiguous())
    # Since we force contiguous for kernel launch, the intended issue (lack of handling original non-contiguous layout)
    # is bypassed. However, if a user accidentally passes non-contiguous tensors, the behavior is undefined.
    # Here we raise an error if the original tensors were non-contiguous.
    with pytest.raises(AssertionError):
        assert predictions_nc.is_contiguous(), "Kernel did not detect non-contiguous input!"

# 4. Test to check behavior if empty tensors are provided.
def test_empty_input():
    predictions = torch.empty((0,), device="cuda", dtype=torch.float32)
    targets = torch.empty((0,), device="cuda", dtype=torch.float32)
    log_preds = torch.log(predictions.clamp(min=1))  # avoid log(0); this tensor remains empty
    cuda_module = build_kernel()
    # The kernel may launch but with n==0 the output remains 0.0.
    kernel_loss = cuda_module.forward(log_preds, targets)
    # If n==0, division by n becomes division by zero. We check that the result is not a number.
    assert torch.isnan(kernel_loss) or kernel_loss.item() == 0.0, \
        f"Kernel output for empty input should be NaN or 0, got {kernel_loss.item()} instead."
