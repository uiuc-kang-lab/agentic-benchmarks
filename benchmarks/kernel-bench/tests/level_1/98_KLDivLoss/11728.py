
import torch
import pytest
import math
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the CUDA extension module from kernel.cu.
    cuda_module = load(
        name="kl_div_kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

def correct_kl_div(log_predictions, targets, reduction='batchmean'):
    # Computes the expected KL divergence as implemented by
    # torch.nn.functional.kl_div:
    # loss = sum(target * (log(target) - log_predictions)) / divisor,
    # where divisor is the batch size for reduction='batchmean'.
    loss = (targets * (torch.log(targets) - log_predictions)).sum()
    if reduction == 'batchmean':
        batch_size = log_predictions.shape[0]
        loss = loss / batch_size
    return loss

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_incorrect_formula():
    # Test 1: Inputs are chosen such that predictions and targets are identical uniform distributions.
    # For correct KL divergence, the loss should be 0.
    # However, since the kernel omits the target*log(target) term, it will produce a nonzero value.
    batch_size = 4
    num_elements = 8  # Divisible by 4 to avoid alignment/remainder issues.
    uniform_val = 1.0 / num_elements
    predictions = torch.full((batch_size, num_elements), uniform_val, device="cuda", dtype=torch.float32)
    targets = predictions.clone()
    
    # Using torch.log, create the input expected by our CUDA kernel.
    log_predictions = torch.log(predictions)
    # Expected loss: when predictions == targets, KL divergence should be 0.
    ref_loss = correct_kl_div(log_predictions, targets).item()
    
    kernel_module = build_kernel()
    kernel_loss = kernel_module.forward(log_predictions, targets).item()
    
    # Check that the kernel loss is not close to 0.
    assert not math.isclose(kernel_loss, ref_loss, rel_tol=1e-3), \
        f"Kernel did not trigger the formula issue: kernel_loss={kernel_loss} but expected {ref_loss}"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_incorrect_normalization():
    # Test 2: Use a batch with a number of elements so that the incorrect normalization (dividing by total n)
    # will differ from the expected 'batchmean' division by batch_size.
    batch_size = 8
    num_elements = 16  # Divisible by 4.
    predictions = torch.softmax(torch.randn(batch_size, num_elements, device="cuda", dtype=torch.float32), dim=-1)
    targets = torch.softmax(torch.randn(batch_size, num_elements, device="cuda", dtype=torch.float32), dim=-1)
    log_predictions = torch.log(predictions)
    
    ref_loss = correct_kl_div(log_predictions, targets).item()
    
    kernel_module = build_kernel()
    kernel_loss = kernel_module.forward(log_predictions, targets).item()
    
    # Expected behavior: if the kernel divided by total number of elements (n = batch_size*num_elements)
    # then kernel_loss would equal ref_loss / num_elements.
    expected_kernel_loss = ref_loss / num_elements

    # The kernel_loss should differ from the ref_loss computed by torch.nn.functional.kl_div.
    assert not math.isclose(kernel_loss, ref_loss, rel_tol=1e-3), \
        f"Kernel did not trigger the normalization issue: kernel_loss={kernel_loss} vs ref_loss={ref_loss}"
    # Also, check that the observed kernel_loss approximates ref_loss/num_elements.
    assert math.isclose(kernel_loss, expected_kernel_loss, rel_tol=1e-3), \
        f"Kernel normalization error: got kernel_loss={kernel_loss} but expected approximately {expected_kernel_loss}"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_alignment_issue():
    # Test 3: Create input tensors with a number of elements NOT divisible by 4
    # to trigger potential misalignment issues with float4 loads.
    batch_size = 4
    num_elements = 10  # 10 is not divisible by 4.
    predictions = torch.softmax(torch.randn(batch_size, num_elements, device="cuda", dtype=torch.float32), dim=-1)
    targets = torch.softmax(torch.randn(batch_size, num_elements, device="cuda", dtype=torch.float32), dim=-1)
    log_predictions = torch.log(predictions)
    
    kernel_module = build_kernel()
    try:
        loss = kernel_module.forward(log_predictions, targets)
        torch.cuda.synchronize()
    except Exception as e:
        pytest.fail(f"Kernel failed with misaligned input tensor (num_elements not divisible by 4): {e}")
