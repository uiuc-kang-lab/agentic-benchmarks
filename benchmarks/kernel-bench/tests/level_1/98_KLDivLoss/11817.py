
import torch
import pytest
from torch.utils.cpp_extension import load
import os

# Helper to build the CUDA kernel module.
def build_kernel():
    # Use a unique name to avoid caching issues.
    module = load(
        name="kl_div_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

# Test 1: Detect Incorrect KL Divergence Formula.
def test_incorrect_formula():
    # Create input distributions with a known outcome.
    # We construct predictions and targets such that the correct kl_div (torch version)
    # can be computed easily.
    torch.manual_seed(42)
    batch_size = 10
    feature_size = 128
    # Use softmax so they are valid probability distributions.
    predictions = torch.rand(batch_size, feature_size, device="cuda").softmax(dim=-1)
    targets = torch.rand(batch_size, feature_size, device="cuda").softmax(dim=-1)
    # Expected result computed via torch.nn.functional.kl_div
    expected = torch.nn.functional.kl_div(torch.log(predictions), targets, reduction='batchmean')
    
    my_module = build_kernel()
    # Kernel forward expects flat tensors.
    log_predictions = torch.log(predictions).contiguous()
    targets_flat = targets.contiguous()
    # Invoke kernel.
    out = my_module.forward(log_predictions, targets_flat)
    torch.cuda.synchronize()
    
    # Since the kernel computes a different formula,
    # we expect a significant difference.
    assert not torch.allclose(out, expected, atol=1e-5), \
        f"Kernel output unexpectedly matches the expected KL divergence. Output: {out} Expected: {expected}"

# Test 2: Trigger Constant Memory Indexing Issue.
def test_constant_memory_indexing():
    # Create a tensor with n less than the constant memory capacity (e.g., n=500)
    # but the kernel launch still spawns more than n threads.
    n = 500  # less than 8192 -> constant memory branch is used.
    # Create a flat tensor.
    log_predictions = torch.randn(n, device="cuda")
    targets = torch.rand(n, device="cuda").abs()  # positive values
    
    # Copy to constant memory would use n elements. However, the kernel does not check n in the constant branch.
    my_module = build_kernel()
    out = my_module.forward(log_predictions, targets)
    torch.cuda.synchronize()
    
    # Compute a reference using a serial loop on the device (simulate correct behavior)
    # Here we compute the kernel's intended per-element operation correctly and average.
    # NOTE: This reference does not include the missing term from the actual formula but at least it uses n.
    correct_vals = torch.empty(n, device="cuda")
    for i in range(n):
        correct_vals[i] = torch.exp(log_predictions[i]) - targets[i] * log_predictions[i]
    expected = correct_vals.sum() / n
    
    # With the constant memory bug, the output is likely to diverge from the expected per-element computation.
    assert not torch.allclose(out, expected, atol=1e-5), \
        f"Kernel output unexpectedly matches a correctly indexed computation. Output: {out} Expected: {expected}"

# Test 3: Hard-coded Block Reduction Assumption.
def test_hardcoded_block_reduction():
    # Create an input size that is not a multiple of 256,
    # which could expose issues in the block reduction that assumes 256 threads per block.
    n = 300  # not a multiple of warp or block sizes
    log_predictions = torch.randn(n, device="cuda")
    targets = torch.rand(n, device="cuda").abs()
    
    my_module = build_kernel()
    out = my_module.forward(log_predictions, targets)
    torch.cuda.synchronize()
    
    # Compute a reference reduction in a straightforward loop:
    correct_vals = torch.empty(n, device="cuda")
    for i in range(n):
        correct_vals[i] = torch.exp(log_predictions[i]) - targets[i] * log_predictions[i]
    expected_reduction = correct_vals.sum() / n
    
    # If the reduction in the kernel is hard-coded, the result may be off.
    assert not torch.allclose(out, expected_reduction, atol=1e-5), \
        f"Kernel block reduction appears to work unexpectedly. Output: {out} Expected: {expected_reduction}"

# Test 4: Incorrect Final Normalization (Division Factor).
def test_incorrect_normalization():
    # Use a 2D tensor to compute batchmean (which should be divided by batch size)
    batch_size = 16
    feature_size = 256
    predictions = torch.rand(batch_size, feature_size, device="cuda").softmax(dim=-1)
    targets = torch.rand(batch_size, feature_size, device="cuda").softmax(dim=-1)
    
    # Expected using torch.nn.functional.kl_div (which divides by batch_size)
    expected = torch.nn.functional.kl_div(torch.log(predictions), targets, reduction='batchmean')
    
    my_module = build_kernel()
    log_predictions = torch.log(predictions).contiguous().view(-1)
    targets_flat = targets.contiguous().view(-1)
    out = my_module.forward(log_predictions, targets_flat)
    torch.cuda.synchronize()
    
    # The kernel divides by total number of elements instead of batch_size.
    # They will differ unless batch_size equals number_of_elements.
    assert not torch.allclose(out, expected, atol=1e-5), \
        f"Kernel normalization unexpectedly matches batchmean. Kernel output: {out} Expected: {expected}"
        
if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])
