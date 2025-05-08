
import pytest
import torch
from torch.utils.cpp_extension import load

# Utility to build and load the CUDA extension from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="kl_div_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# -----------------------------------------------------------------------------
# 1. Test to trigger the incorrect KL divergence formula.
# This test sets up a scenario where the expected KL divergence can be computed via PyTorch's in‑built function.
# We expect the kernel result (which uses an incorrect formula) to differ from the PyTorch result.
def test_incorrect_formula():
    # Create inputs in float32 on CUDA.
    batch_size, n_features = 128, 4096
    predictions = torch.randn(batch_size, n_features, device="cuda", dtype=torch.float32)
    targets = torch.randn(batch_size, n_features, device="cuda", dtype=torch.float32)
    # Apply softmax to create probability distributions.
    predictions = torch.softmax(predictions, dim=-1)
    targets = torch.softmax(targets, dim=-1)

    # Compute our expected result using PyTorch's torch.nn.functional.kl_div.
    log_predictions = torch.log(predictions)
    expected = torch.nn.functional.kl_div(log_predictions, targets, reduction='batchmean')
    
    module = build_kernel()
    # The kernel expects its first argument to be log_predictions
    result = module.forward(log_predictions.contiguous(), targets.contiguous())
    
    # Since the math is wrong in the kernel, the result should differ.
    # We check that they are NOT almost equal.
    assert not torch.allclose(result, expected, atol=1e-4), \
        f"Kernel result unexpectedly matches PyTorch result although the formula is implemented incorrectly! Kernel: {result.item()}, Expected: {expected.item()}"

# -----------------------------------------------------------------------------
# 2. Test to trigger the incorrect reduction factor.
# The kernel returns sum/n, but for 'batchmean', the sum should be divided by batch size.
# This test sets up inputs where batch_size != total elements so the reduction is clearly off.
def test_incorrect_reduction():
    # Use a batch size that is much smaller than the total number of elements per sample.
    batch_size, n_features = 8, 4096  # Total elements = 8*4096
    predictions = torch.randn(batch_size, n_features, device="cuda", dtype=torch.float32)
    targets = torch.randn(batch_size, n_features, device="cuda", dtype=torch.float32)
    predictions = torch.softmax(predictions, dim=-1)
    targets = torch.softmax(targets, dim=-1)

    log_predictions = torch.log(predictions)
    expected = torch.nn.functional.kl_div(log_predictions, targets, reduction='batchmean')
    
    module = build_kernel()
    result = module.forward(log_predictions.contiguous(), targets.contiguous())
    
    # Given that the kernel divides by the total number of elements instead of batch_size,
    # when batch_size < (batch_size * n_features) the result will be smaller than expected.
    assert not torch.allclose(result, expected, atol=1e-4), \
        f"Kernel reduction did not reveal the error in division factor. Kernel: {result.item()}, Expected: {expected.item()}"

# -----------------------------------------------------------------------------
# 3. Test to trigger errors when using a tensor with an unsupported data type (e.g. float64).
def test_input_tensor_type():
    batch_size, n_features = 128, 4096
    # Create double precision inputs.
    predictions = torch.randn(batch_size, n_features, device="cuda", dtype=torch.float64)
    targets = torch.randn(batch_size, n_features, device="cuda", dtype=torch.float64)
    predictions = torch.softmax(predictions, dim=-1)
    targets = torch.softmax(targets, dim=-1)
    log_predictions = torch.log(predictions)
    
    module = build_kernel()
    # Expect a RuntimeError or wrong behavior if double precision is passed.
    with pytest.raises(RuntimeError):
        # This should raise an error because the kernel expects float32.
        module.forward(log_predictions.contiguous(), targets.contiguous())

# -----------------------------------------------------------------------------
# 4. Test to trigger issues with non-contiguous memory accesses.
# The kernel assumes contiguous and properly aligned memory for vectorized loads.
def test_non_contiguous():
    batch_size, n_features = 128, 4096
    predictions = torch.randn(batch_size, n_features, device="cuda", dtype=torch.float32)
    targets = torch.randn(batch_size, n_features, device="cuda", dtype=torch.float32)
    predictions = torch.softmax(predictions, dim=-1)
    targets = torch.softmax(targets, dim=-1)
    
    # Make tensors non-contiguous by transposing a 2D version (and then taking a view).
    log_predictions = torch.log(predictions)
    log_predictions_nc = log_predictions.t()  # Transpose makes it non-contiguous
    targets_nc = targets.t()
    
    module = build_kernel()
    # Depending on alignment assumptions, the kernel may produce wrong results or crash.
    # We catch any exception as a sign to indicate the issue.
    with pytest.raises(Exception):
        module.forward(log_predictions_nc, targets_nc)

