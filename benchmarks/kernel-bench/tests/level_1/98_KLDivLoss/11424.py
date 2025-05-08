
import torch
import pytest
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

# Helper: compute reference KL divergence using PyTorch's kl_div function
def compute_reference_kl(log_predictions, targets):
    # F.kl_div expects log_predictions and targets and performs: target*(log(target) - log_prediction)
    # with 'batchmean' reduction (i.e. sum over all elements divided by batch size)
    batch_size = log_predictions.shape[0]
    # Avoiding log(0) by adding a small epsilon.
    eps = 1e-12
    kl = targets * (torch.log(targets + eps) - log_predictions)
    # Sum over non-batch dims, average over batch.
    kl = kl.reshape(batch_size, -1).sum(dim=1).mean()
    return kl

def test_math_formula():
    """
    This test is designed to expose the kernel's incorrect formula.
    The expected behaviour (from torch.nn.functional.kl_div) is different from what the kernel computes.
    Even if the relative error is moderate in some cases, for controlled inputs the discrepancy should be evident.
    """
    torch.manual_seed(0)
    batch_size, dim = 16, 1024
    # Create proper probability distributions (softmax)
    predictions = torch.randn(batch_size, dim, device='cuda').softmax(dim=-1)
    targets = torch.randn(batch_size, dim, device='cuda').softmax(dim=-1)
    # Compute log_predictions input for the kernel (which expects log probabilities)
    log_predictions = predictions.log()
    
    my_module = build_kernel()
    # Run the custom kernel
    kernel_out = my_module.forward(log_predictions, targets)
    # Compute reference kl divergence using pytorch's F.kl_div
    ref_out = torch.nn.functional.kl_div(log_predictions, targets, reduction='batchmean')
    
    # The outputs are expected to significantly differ because the kernel omits target*log(target)
    # The test will fail if they are too close.
    diff = (kernel_out - ref_out).abs().item()
    assert diff > 1e-3, f"Test did not expose formula difference. Diff: {diff}"

def test_reduction_scaling():
    """
    This test checks that the kernel output scaling is performed incorrectly.
    The kernel divides by n (total elements) whereas PyTorch's 'batchmean' divides by batch size.
    For different batch sizes, this discrepancy should become evident.
    """
    torch.manual_seed(0)
    # Use a larger batch to see the scaling difference
    batch_size, dim = 64, 2048
    predictions = torch.randn(batch_size, dim, device='cuda').softmax(dim=-1)
    targets = torch.randn(batch_size, dim, device='cuda').softmax(dim=-1)
    log_predictions = predictions.log()
    
    my_module = build_kernel()
    kernel_out = my_module.forward(log_predictions, targets)
    ref_out = torch.nn.functional.kl_div(log_predictions, targets, reduction='batchmean')
    
    # In a correct implementation, kernel_out and ref_out would be nearly equal.
    # We expect a significant scaling difference because of division by total number of elements vs batch size.
    diff = (kernel_out - ref_out).abs().item()
    assert diff > 1e-3, f"Reduction scaling issue not detected. Diff: {diff}"

def test_input_type():
    """
    This test provides double (float64) tensors as input.
    Since the kernel only supports float32, the computation is expected to be incorrect or raise an error.
    """
    torch.manual_seed(0)
    batch_size, dim = 16, 512
    predictions = torch.randn(batch_size, dim, device='cuda', dtype=torch.float64).softmax(dim=-1)
    targets = torch.randn(batch_size, dim, device='cuda', dtype=torch.float64).softmax(dim=-1)
    log_predictions = predictions.log()  # remains float64
    
    my_module = build_kernel()
    # We expect that running the kernel with float64 inputs will either throw an error
    # or produce a result that is inconsistent with float32 computations.
    with pytest.raises(RuntimeError):
        # If the kernel attempts to treat the data as float32, it should trigger an error.
        my_module.forward(log_predictions, targets)

def test_input_dim():
    """
    This test passes non-flat, non-contiguous tensors to the kernel.
    The kernel assumes contiguous 1D arrays, so using multidimensional or non-contiguous inputs should reveal potential issues.
    """
    torch.manual_seed(0)
    # Create a 3D tensor (for example, batch x channels x length) that is not flattened
    batch_size, channels, length = 8, 4, 1024
    predictions = torch.randn(batch_size, channels, length, device='cuda').softmax(dim=-1)
    targets = torch.randn(batch_size, channels, length, device='cuda').softmax(dim=-1)
    log_predictions = predictions.log()
    
    my_module = build_kernel()
    # Since the kernel internally uses numel() and expects a flat layout corresponding directly to the probability vector,
    # the results will not match the expected result from torch.nn.functional.kl_div.
    kernel_out = my_module.forward(log_predictions, targets)
    ref_out = torch.nn.functional.kl_div(log_predictions, targets, reduction='batchmean')
    
    diff = (kernel_out - ref_out).abs().item()
    assert diff > 1e-3, f"Expected dimensionality issues were not detected (diff: {diff})"

def test_min_function():
    """
    This test does not directly call a runtime error but is designed to highlight a potential compilation issue.
    The use of the unqualified 'min' in the host code may cause compilation problems.
    If the module is successfully built, this test passes.
    """
    try:
        my_module = build_kernel()
        # If build_kernel() works then the issue with 'min' did not block compilation.
        assert hasattr(my_module, "forward"), "Module does not export function 'forward'"
    except Exception as e:
        pytest.fail(f"Compilation issue related to unqualified min: {e}")
