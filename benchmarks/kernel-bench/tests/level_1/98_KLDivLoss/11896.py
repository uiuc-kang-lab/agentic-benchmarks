
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    # This call compiles the CUDA extension.
    cuda_module = load(
        name="kl_div_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

def correct_kl_div(log_predictions, targets, reduction='batchmean'):
    # Return the PyTorch reference KL-divergence computed with torch.nn.functional.kl_div
    return torch.nn.functional.kl_div(log_predictions, targets, reduction=reduction)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_incorrect_math():
    """
    Test case to demonstrate the incorrect mathematical formulation.
    The reference KL divergence (using torch.nn.functional.kl_div) will differ from the custom kernel’s output.
    """
    kernel = build_kernel()
    # Use a small, controlled example.
    predictions = torch.tensor([[0.1, 0.9]], device='cuda', dtype=torch.float32)
    targets = torch.tensor([[0.2, 0.8]], device='cuda', dtype=torch.float32)
    log_predictions = torch.log(predictions)
    
    out_cuda = kernel.forward(log_predictions, targets)
    out_torch = correct_kl_div(log_predictions, targets)
    
    # The outputs should differ because the kernel computes an incorrect formula.
    assert not torch.allclose(out_cuda, out_torch, atol=1e-5), (
        f"Kernel incorrectly matched reference output: kernel {out_cuda.item()} vs torch {out_torch.item()}"
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_normalization_factor():
    """
    Test case to expose the incorrect normalization.
    The kernel divides by total number of elements, while 'batchmean' reduction divides only by the batch size.
    """
    kernel = build_kernel()
    # Create a batched input where batch dimension != number of elements.
    predictions = torch.softmax(torch.randn(4, 3, device='cuda', dtype=torch.float32), dim=1)
    targets = torch.softmax(torch.randn(4, 3, device='cuda', dtype=torch.float32), dim=1)
    log_predictions = torch.log(predictions)
    
    out_cuda = kernel.forward(log_predictions, targets)
    out_torch = correct_kl_div(log_predictions, targets)  # Reference output with batchmean normalization.
    
    # Expect a discrepancy in normalization.
    assert not torch.allclose(out_cuda, out_torch, atol=1e-5), (
        f"Kernel normalization factor appears correct unexpectedly: kernel {out_cuda.item()} vs torch {out_torch.item()}"
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_workload_distribution():
    """
    Test case to trigger potential issues with workload distribution.
    The grid configuration uses ELEMENTS_PER_THREAD in its calculation, but the actual loop does not use it,
    which may lead to problems when the number of elements is not perfectly divisible.
    """
    kernel = build_kernel()
    # Input with a shape that results in a total element count not divisible by BLOCK_SIZE * ELEMENTS_PER_THREAD.
    # For example, using 17 columns instead of a multiple of 4 (if ELEMENTS_PER_THREAD==4).
    predictions = torch.softmax(torch.randn(128, 17, device='cuda', dtype=torch.float32), dim=1)
    targets = torch.softmax(torch.randn(128, 17, device='cuda', dtype=torch.float32), dim=1)
    log_predictions = torch.log(predictions)
    
    out_cuda = kernel.forward(log_predictions, targets)
    out_torch = correct_kl_div(log_predictions, targets)
    
    # Due to unexpected workload distribution, the kernel output will not match the reference.
    assert not torch.allclose(out_cuda, out_torch, atol=1e-5), (
        f"Kernel workload distribution appears correct unexpectedly: kernel {out_cuda.item()} vs torch {out_torch.item()}"
    )
