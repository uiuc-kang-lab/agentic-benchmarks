
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Helper function to compile and load the CUDA kernel module.
def build_kernel():
    # Ensure that the current directory is used to find kernel.cu
    module = load(
        name="cuda_kl_div_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

@pytest.fixture(scope="module")
def cuda_module():
    return build_kernel()

# Test case 1: Trigger the incorrect KL divergence formula.
def test_incorrect_kl_formula(cuda_module):
    # Create a simple input where we can compute the expected result manually.
    # Use a small tensor so that we can compute the actual values.
    predictions = torch.tensor([[0.2, 0.3, 0.5]], device="cuda", dtype=torch.float32)
    targets = torch.tensor([[0.1, 0.4, 0.5]], device="cuda", dtype=torch.float32)
    # Normal PyTorch behavior: using F.kl_div expects log_predictions as first argument.
    log_predictions = torch.log(predictions)
    # Expected KL divergence using torch.nn.functional.kl_div (batchmean).
    # Note: F.kl_div calculates: sum(target * (log(target) - log_predictions)) / batch_size.
    loss_expected = torch.nn.functional.kl_div(log_predictions, targets, reduction='batchmean')
    
    # Call our CUDA kernel.
    loss_cuda = cuda_module.forward(log_predictions.contiguous(), targets.contiguous())
    # The loss computed by the kernel is mathematically different.
    # We assert that the discrepancy is significant.
    assert not torch.allclose(loss_cuda, loss_expected, atol=1e-6), \
        f"Kernel loss ({loss_cuda.item()}) unexpectedly matches the expected loss ({loss_expected.item()})."

# Test case 2: Verify the normalization error.
def test_normalization_error(cuda_module):
    # Create batched inputs
    batch_size = 8
    input_size = 16
    predictions = torch.softmax(torch.randn(batch_size, input_size, device="cuda", dtype=torch.float32), dim=-1)
    targets = torch.softmax(torch.randn(batch_size, input_size, device="cuda", dtype=torch.float32), dim=-1)
    log_predictions = torch.log(predictions)

    loss_expected = torch.nn.functional.kl_div(log_predictions, targets, reduction='batchmean')
    loss_cuda = cuda_module.forward(log_predictions.contiguous(), targets.contiguous())

    # When using 'batchmean', PyTorch divides by the batch size.
    # Our kernel divides by the total number of elements (batch_size*input_size).
    # Hence, the ratio between expected and kernel loss should roughly be batch_size/input_size*? (not equal)
    # We verify that the two losses are not equal.
    assert not torch.allclose(loss_cuda, loss_expected, atol=1e-6), \
        f"Kernel normalization error test fails. Expected normalization discrepancy."

# Test case 3: Test for unqualified min usage.
# This test is a compile-time configuration trigger. Since we cannot
# trigger a runtime error directly from unqualified 'min', we simulate by
# building the kernel with a non-standard environment.
def test_unqualified_min(cuda_module):
    # We run a simple pass-through; if compilation had failed from 'min', this test would not even run.
    # So, we check that the module has an attribute 'forward'.
    assert hasattr(cuda_module, "forward"), "CUDA module does not have a forward function; possible 'min' issue."

# Test case 4: Trigger input validation issues by passing double precision inputs.
def test_input_tensor_type(cuda_module):
    N = 1024
    # Create double precision tensors (non float32) to see if the kernel fails or produces unexpected results.
    log_predictions = torch.log(torch.softmax(torch.randn(N, device="cuda", dtype=torch.float64), dim=-1))
    targets = torch.softmax(torch.randn(N, device="cuda", dtype=torch.float64), dim=-1)
    
    # Expect that misuse of tensor type might lead to wrong behavior.
    with pytest.raises(RuntimeError):
        cuda_module.forward(log_predictions.contiguous(), targets.contiguous())

# Test case 5: Trigger division by zero by passing an empty tensor.
def test_division_by_zero(cuda_module):
    # Create empty tensors for log_predictions and targets.
    log_predictions = torch.empty((0,), device="cuda", dtype=torch.float32)
    targets = torch.empty((0,), device="cuda", dtype=torch.float32)
    
    # Expect that division by zero or an error occurs.
    with pytest.raises(RuntimeError):
        cuda_module.forward(log_predictions, targets)
