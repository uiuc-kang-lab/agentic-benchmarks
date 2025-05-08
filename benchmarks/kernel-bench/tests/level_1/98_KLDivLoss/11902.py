
import torch
import pytest
from torch.utils.cpp_extension import load

# Utility: compile and load the CUDA kernel from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Incorrect computation (wrong mathematical formula).
# For controlled inputs we can set a known simple distribution.
# Here we use a small tensor where we can compute the expected KL divergence manually.
def test_incorrect_formula():
    # Create a simple probability distribution on a single distribution.
    # We use one row (batch_size=1) and a few columns.
    # For input to the kernel, PyTorch's kl_div expects log predictions.
    predictions = torch.tensor([[0.1, 0.2, 0.7]], dtype=torch.float32, device='cuda')
    log_predictions = torch.log(predictions)
    # Let targets be another distribution.
    targets = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float32, device='cuda')

    # Expected KL divergence as computed by PyTorch's built-in function.
    # Note: torch.nn.functional.kl_div expects log_predictions as first argument.
    expected = torch.nn.functional.kl_div(log_predictions, targets, reduction='batchmean')

    module = build_kernel()
    # Call the kernel using our compiled module.
    result = module.forward(log_predictions, targets)
    torch.cuda.synchronize()

    # The kernel computes a different value because it uses an incorrect formula.
    # We expect the kernel output not to match the correct KL divergence.
    # Here we trigger the issue by asserting they are not approximately equal.
    assert not torch.allclose(result, expected, atol=1e-5), \
        f"Kernel unexpectedly computed the correct value; result = {result}, expected = {expected}"

# Issue 2: Incorrect reduction (normalization by total elements rather than batch_size).
def test_wrong_reduction():
    # Use a batch of two distributions where the difference in reduction will be clear.
    batch_size = 2
    input_dim = 4
    # Create arbitrary distributions and compute log predictions.
    predictions = torch.full((batch_size, input_dim), 0.25, dtype=torch.float32, device='cuda')
    log_predictions = torch.log(predictions)
    # Choose a target that is one-hot for easier manual reasoning.
    targets = torch.zeros((batch_size, input_dim), dtype=torch.float32, device='cuda')
    targets[0, 1] = 1.0
    targets[1, 2] = 1.0

    # Compute expected value using PyTorch function with reduction='batchmean'
    expected = torch.nn.functional.kl_div(log_predictions, targets, reduction='batchmean')
    module = build_kernel()
    result = module.forward(log_predictions, targets)
    torch.cuda.synchronize()

    # The kernel divides by total number of elements (batch_size * input_dim)
    # rather than by the batch_size, so the result will be scaled by (1/input_dim)
    # relative to the expected value.
    scaled_expected = expected * (1.0 / input_dim)
    assert torch.allclose(result, scaled_expected, atol=1e-5), \
        f"Kernel reduction issue: result = {result} does not match scaled expected = {scaled_expected}"

# Issue 3: Kernel does not support double precision inputs.
def test_input_dtype():
    # Create a simple input in double precision.
    batch_size = 16
    input_dim = 8
    predictions = torch.randn(batch_size, input_dim, dtype=torch.float64, device='cuda').softmax(dim=-1)
    # Log predictions
    log_predictions = torch.log(predictions)
    targets = torch.randn(batch_size, input_dim, dtype=torch.float64, device='cuda').softmax(dim=-1)

    module = build_kernel()
    # The kernel explicitly uses float pointers and expf(). Using double tensors
    # may either cast silently or produce nonsensical results.
    # We trigger the issue by checking that the output is not equal to the expected float32 computation,
    # or that an error is raised (depending on the implementation details).
    with pytest.raises(RuntimeError):
        # This test expects that an error will be thrown due to data type mismatch.
        _ = module.forward(log_predictions, targets)
