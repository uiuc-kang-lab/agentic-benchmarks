
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

@pytest.fixture(scope="module")
def kernel_module():
    return build_kernel()

# Issue 1: Incorrect mathematical formulation.
# For identical predictions and targets the KL divergence should be zero.
def test_incorrect_math(kernel_module):
    # Create a tensor where predictions == targets.
    # For a proper KL divergence computation, if P==Q then KL = 0.
    # The kernel calls expect log(predictions) and targets.
    batch_size = 128
    features = 4096
    # Create predictions = targets (softmax ensures valid distribution)
    preds = torch.rand(batch_size, features, device="cuda", dtype=torch.float32)
    preds = torch.softmax(preds, dim=-1)
    # Make targets identical to predictions.
    targets = preds.clone()

    # Compute log_predictions for kernel.
    log_preds = torch.log(preds)
    # Run the kernel.
    output = kernel_module.forward(log_preds, targets)
    torch.cuda.synchronize()
    
    # Since KL divergence should be zero, any nonzero output reveals the math error.
    # We expect a significant non-zero value because the formula is wrong.
    assert torch.abs(output).item() > 1e-3, f"Expected nonzero output due to wrong math, but got {output.item()}"

# Issue 2: Inconsistent reduction scaling.
# Compare the kernel output with torch.nn.functional.kl_div on the same inputs.
def test_inconsistent_scaling(kernel_module):
    batch_size = 128
    features = 4096
    # Create valid distributions.
    predictions = torch.rand(batch_size, features, device="cuda", dtype=torch.float32)
    predictions = torch.softmax(predictions, dim=-1)
    
    targets = torch.rand(batch_size, features, device="cuda", dtype=torch.float32)
    targets = torch.softmax(targets, dim=-1)
    
    # Compute log_predictions for kernel.
    log_preds = torch.log(predictions)
    
    # Compute kernel output.
    output_kernel = kernel_module.forward(log_preds, targets)
    torch.cuda.synchronize()
    
    # Compute reference KL divergence using PyTorch’s implementation.
    # Note: torch.nn.functional.kl_div uses reduction='batchmean'
    output_ref = torch.nn.functional.kl_div(torch.log(predictions), targets, reduction='batchmean')

    # The kernel divides by total number of elements, while reference divides by batch size.
    # Hence, the kernel output, if computed correctly element‐wise, would be off by a factor of (features).
    # We expect the error ~ factor features if the math were right.
    scaling_factor = features
    diff = torch.abs(output_kernel - output_ref * scaling_factor).item()
    assert diff > 1e-3, f"Kernel scaling appears to be correct, but an inconsistency is expected. Difference: {diff}"

# Issue 3: Lack of data type checks.
# Provide double tensors to trigger the mistaken assumption of float32.
def test_incorrect_dtype(kernel_module):
    batch_size = 128
    features = 4096
    # Create double precision distributions.
    predictions = torch.rand(batch_size, features, device="cuda", dtype=torch.float64)
    predictions = torch.softmax(predictions, dim=-1)
    
    targets = torch.rand(batch_size, features, device="cuda", dtype=torch.float64)
    targets = torch.softmax(targets, dim=-1)
    
    # Compute log_predictions for kernel.
    # Cast explicitly to float64 to simulate what the user would do.
    log_preds = torch.log(predictions)
    
    # Call the kernel with double tensors.
    # The kernel expects float (float32), so the resulting calculation will be incorrect.
    output_kernel = kernel_module.forward(log_preds, targets)
    torch.cuda.synchronize()
    
    # Compute reference KL divergence using double precision.
    output_ref = torch.nn.functional.kl_div(torch.log(predictions), targets, reduction='batchmean')
    
    # They will differ because the kernel incorrectly casts or misinterprets the data.
    diff = torch.abs(output_kernel - output_ref).item()
    assert diff > 1e-3, f"Kernel should fail to process double precision correctly, but difference: {diff}"
