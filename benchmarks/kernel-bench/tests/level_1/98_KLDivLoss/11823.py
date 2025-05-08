
import torch
import pytest
from torch.utils.cpp_extension import load
import math

def build_kernel():
    cuda_module = load(
        name="kl_div_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Helper function to compute the “correct” KL divergence value as per PyTorch's kl_div
# Here, for an element, the expected value is: target*(log(target) - log_prediction)
def reference_kl_div(log_predictions, targets, reduction='none'):
    # We assume log_predictions are computed as torch.log(predictions)
    vals = targets * (torch.log(targets) - log_predictions)
    if reduction == 'batchmean':
        batch_size = targets.shape[0]
        return vals.sum() / batch_size
    elif reduction == 'mean':
        return vals.mean()
    else:
        return vals.sum()

def test_incorrect_divergence_formula():
    # Test to showcase that the computed divergence from the kernel misses the target*log(target) term.
    # Create simple tensors where we know what to expect.
    torch.manual_seed(0)
    batch_size = 4
    features = 16  # choose a number divisible by 4 for simplicity
    # Create valid probability distributions via softmax, and then take log for the kernel input.
    preds = torch.rand(batch_size, features, device='cuda').softmax(dim=-1)
    targets = torch.rand(batch_size, features, device='cuda').softmax(dim=-1)
    log_preds = torch.log(preds)
    
    # Get reference result using the correct formula (batchmean reduction)
    ref = reference_kl_div(log_preds, targets, reduction='batchmean')
    
    kernel_module = build_kernel()
    # Kernel function expects two float tensors
    result = kernel_module.forward(log_preds.contiguous(), targets.contiguous())
    torch.cuda.synchronize()
    
    # Because the kernel is missing target*log(target), the difference should be nonzero.
    diff = abs(result.item() - ref.item())
    assert diff > 1e-3, f"Kernel result unexpectedly matches reference KL divergence. Diff: {diff}"

def test_wrong_normalization():
    # Test to reveal that the kernel divides by n (total number of elements) instead of the batch size.
    torch.manual_seed(0)
    batch_size = 8
    features = 32  # divisible by 4 here
    preds = torch.rand(batch_size, features, device='cuda').softmax(dim=-1)
    targets = torch.rand(batch_size, features, device='cuda').softmax(dim=-1)
    log_preds = torch.log(preds)
    
    kernel_module = build_kernel()
    result = kernel_module.forward(log_preds.contiguous(), targets.contiguous())
    torch.cuda.synchronize()
    
    # The kernel divides by total number of elements (batch_size*features) rather than batch_size.
    # So if we multiply the kernel result by features we expect approximately the same sum as
    # computed without dividing by "features". This test shows that the normalization is off.
    kernel_sum = result.item() * (batch_size * features)
    # Compute reference sum using the flawed kernel's partial reduction: we expect that 
    # the correct batchmean would be ref_sum / batch_size but kernel does ref_sum / (batch_size * features)
    # Hence, kernel_sum should differ from the reference sum computed using correct batchmean scaling.
    ref = reference_kl_div(log_preds, targets, reduction='none').sum().item()
    diff = abs(kernel_sum - ref)
    assert diff > 1e-3, f"Kernel normalization error not detected. Diff: {diff}"

def test_type_mismatch():
    # Test to verify that passing double precision input leads to an incorrect calculation.
    # The kernel assumes float (32-bit) precision.
    torch.manual_seed(0)
    batch_size = 4
    features = 64
    # Create double tensors instead of float
    preds = torch.rand(batch_size, features, device='cuda', dtype=torch.double).softmax(dim=-1)
    targets = torch.rand(batch_size, features, device='cuda', dtype=torch.double).softmax(dim=-1)
    log_preds = torch.log(preds)
    
    kernel_module = build_kernel()
    # Expect that the kernel (compiled for float32) will not work correctly when given double tensors.
    # We wrap the call in pytest.raises to catch any exception indicating type misuse.
    with pytest.raises(RuntimeError):
        # This should raise an error or produce invalid results due to type mismatch.
        _ = kernel_module.forward(log_preds.contiguous(), targets.contiguous())
    # Alternatively, if no error is raised, we can check that the output is far off.
    # (Uncomment the code below if you prefer a numerical check instead of exception.)
    # result = kernel_module.forward(log_preds.contiguous(), targets.contiguous())
    # torch.cuda.synchronize()
    # assert result.item() != result.item(), "Kernel accepted double precision input incorrectly."

def test_non_divisible_by_four():
    # Test to expose potential issues when the number of elements is not divisible by 4.
    torch.manual_seed(0)
    # Create a tensor whose last dimension is such that total element count is not a multiple of 4.
    batch_size = 3
    features = 10  # 3*10 = 30, and 30 mod 4 != 0
    preds = torch.rand(batch_size, features, device='cuda').softmax(dim=-1)
    targets = torch.rand(batch_size, features, device='cuda').softmax(dim=-1)
    log_preds = torch.log(preds)
    
    kernel_module = build_kernel()
    result = kernel_module.forward(log_preds.contiguous(), targets.contiguous())
    torch.cuda.synchronize()
    
    # Compute reference using the flawed kernel formula (ignoring the missing term) for comparison.
    # Here we mimic the kernel's intended computation: sum(exp(log_pred) - target * log_pred) elementwise.
    computed = torch.exp(log_preds) - targets * log_preds
    ref_val = computed.sum().item() / (batch_size * features)
    diff = abs(result.item() - ref_val)
    # Even though the formula is incorrect overall, the processing of tail elements should be triggered.
    assert diff < 1e-3, f"Tail elements not processed correctly when n is not a multiple of 4. Diff: {diff}"

# Pytest entry point
if __name__ == '__main__':
    pytest.main([__file__])
