
import torch
import pytest
from torch.utils.cpp_extension import load
import math

def build_kernel():
    cuda_module = load(
        name="kl_div_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return cuda_module

@pytest.fixture(scope="module")
def kernel_module():
    return build_kernel()

# Test case 1: Incorrect KL divergence formula
def test_incorrect_formula(kernel_module):
    # Use a small tensor so we can compute a reference value manually
    batch_size = 4
    n_elements = 16
    # Create probabilities and add a small constant to avoid log(0)
    predictions = torch.rand(batch_size, n_elements, device='cuda', dtype=torch.float32)
    predictions = predictions.softmax(dim=-1)
    targets = torch.rand(batch_size, n_elements, device='cuda', dtype=torch.float32)
    targets = targets.softmax(dim=-1)
    # Compute reference KL divergence using torch.nn.functional.kl_div.
    # Note: torch.nn.functional.kl_div expects log(input) as first argument.
    preds_log = torch.log(predictions)
    ref = torch.nn.functional.kl_div(preds_log, targets, reduction='batchmean')
    # Use the custom CUDA kernel which is expected to compute a similar value
    out = kernel_module.forward(preds_log, targets)
    # The results are expected to differ because of a missing constant term.
    # We trigger the issue by checking that the error is significant.
    assert not torch.allclose(out, ref, atol=1e-5), (
        "The kernel output is too close to the reference, which is unexpected given the incorrect formula."
    )

# Test case 2: Inconsistent normalization (total elements vs batch size)
def test_normalization_issue(kernel_module):
    batch_size = 8
    n_elements = 32  # choose a size that is not 1024 to avoid constant memory effect in entirety
    predictions = torch.rand(batch_size, n_elements, device='cuda', dtype=torch.float32).softmax(dim=-1)
    targets = torch.rand(batch_size, n_elements, device='cuda', dtype=torch.float32).softmax(dim=-1)
    preds_log = torch.log(predictions)
    # Compute kernel output which divides by total number of elements (batch_size * n_elements)
    out = kernel_module.forward(preds_log, targets)
    # Compute reference using torch.nn.functional.kl_div (which divides by batch_size)
    ref = torch.nn.functional.kl_div(preds_log, targets, reduction='batchmean')
    # They differ by a factor of (n_elements) because kernel divides by (batch_size*n_elements)
    ratio = (batch_size * n_elements) / (batch_size)
    # The kernel output multiplied by the ratio should get closer to the reference if normalization were the only issue.
    corrected = out * ratio
    assert not torch.allclose(corrected, ref, atol=1e-5), (
        "Normalization correction suggests kernel normalization is not using batchmean reduction."
    )

# Test case 3: Constant memory assumption misuse
def test_constant_memory_usage(kernel_module):
    # Create a tensor where the first 1024 values of targets are very different from the rest.
    batch_size = 1
    n_elements = 2048  # more than 1024
    predictions = torch.rand(batch_size, n_elements, device='cuda', dtype=torch.float32).softmax(dim=-1)
    # Set the first 1024 targets to ones, and the rest to zeros (but then renormalize to a probability distribution)
    targets = torch.zeros(batch_size, n_elements, device='cuda', dtype=torch.float32)
    targets[:, :1024] = 1.0
    # Renormalize to form valid probability distributions
    targets = targets.softmax(dim=-1)
    preds_log = torch.log(predictions)
    out = kernel_module.forward(preds_log, targets)
    # Since the kernel reads constant memory for indices < 1024, we expect behavior to differ from a reference computed purely on global memory.
    ref = torch.nn.functional.kl_div(preds_log, targets, reduction='batchmean')
    # We assert that the kernel output is different from the reference.
    assert not torch.allclose(out, ref, atol=1e-5), (
        "Kernel did not exhibit different behavior expected from constant memory usage."
    )

# Test case 4: Data type handling (non-float32 input)
def test_dtype_handling(kernel_module):
    # Create double precision tensors.
    batch_size = 4
    n_elements = 64
    predictions = torch.rand(batch_size, n_elements, device='cuda', dtype=torch.float64).softmax(dim=-1)
    targets = torch.rand(batch_size, n_elements, device='cuda', dtype=torch.float64).softmax(dim=-1)
    preds_log = torch.log(predictions)
    # The kernel expects float* pointers, so using double should either raise an error or produce wrong results.
    with pytest.raises(RuntimeError):
        kernel_module.forward(preds_log, targets)
    
# Test case 5: Empty input handling (n == 0)
def test_empty_input(kernel_module):
    # Create empty tensors that result in n==0.
    predictions = torch.empty((0,), device='cuda', dtype=torch.float32)
    targets = torch.empty((0,), device='cuda', dtype=torch.float32)
    # This is likely to cause a division by zero or launch a kernel with 0 elements.
    with pytest.raises(Exception):
        # Depending on the CUDA runtime, this may raise a runtime error.
        kernel_module.forward(predictions, targets)
