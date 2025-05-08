
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="kl_div_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Trigger the incorrect KL divergence computation.
# Use a simple case where the expected correct KL divergence is known.
def test_kl_div_computation():
    # Create two simple distributions for which torch.nn.functional.kl_div returns 0.
    # For example, if both predictions and targets are identical uniform distributions.
    batch_size = 4
    dims = 8
    predictions = torch.full((batch_size, dims), 1.0, device="cuda", dtype=torch.float32)
    predictions = predictions / predictions.sum(dim=-1, keepdim=True)
    targets = predictions.clone()

    # Compute log(predictions)
    log_predictions = torch.log(predictions)

    # Compute reference using PyTorch's F.kl_div
    ref = torch.nn.functional.kl_div(log_predictions, targets, reduction='batchmean')
    
    # Compute value using the custom kernel.
    kl_div_module = build_kernel()
    custom_out = kl_div_module.forward(log_predictions.contiguous(), targets.contiguous())
    
    # The custom kernel produces a nonzero value due to the missing constant term and wrong scaling.
    # We expect that ref == 0 while custom_out is not.
    assert not torch.isclose(custom_out, ref, atol=1e-5), (
        f"Expected a numerical discrepancy due to incorrect formula and reduction. "
        f"Found kernel output {custom_out.item()} vs ref {ref.item()}"
    )

# Test 2: Trigger the wrong reduction factor issue.
# Provide non-uniform distributions so that scaling differences are evident.
def test_reduction_scaling():
    batch_size = 4
    dims = 8
    # Create a non uniform distribution that sums to 1 over the last dim
    predictions = torch.rand(batch_size, dims, device="cuda", dtype=torch.float32)
    predictions = predictions.softmax(dim=-1)
    targets = torch.rand(batch_size, dims, device="cuda", dtype=torch.float32)
    targets = targets.softmax(dim=-1)

    log_predictions = torch.log(predictions)
    ref = torch.nn.functional.kl_div(log_predictions, targets, reduction='batchmean')
    
    kl_div_module = build_kernel()
    custom_out = kl_div_module.forward(log_predictions.contiguous(), targets.contiguous())
    
    # Due to division by total number of elements in the kernel, the value will be off.
    # We check that the relative difference is significant.
    rel_diff = abs(custom_out.item() - ref.item()) / (abs(ref.item()) + 1e-8)
    assert rel_diff > 0.1, (
        f"Reduction scaling issue: kernel reduction mismatch. Relative difference {rel_diff}"
    )

# Test 3: Trigger type incompatibility issues.
# Provide double precision tensors to see if the kernel (which expects float) may malfunction.
def test_input_tensor_type():
    batch_size = 4
    dims = 8
    predictions = torch.rand(batch_size, dims, device="cuda", dtype=torch.float64)
    predictions = predictions.softmax(dim=-1)
    targets = torch.rand(batch_size, dims, device="cuda", dtype=torch.float64)
    targets = targets.softmax(dim=-1)
    
    # Compute log_predictions in double, but the kernel uses data_ptr<float>()
    # An error or unexpected behavior is expected.
    kl_div_module = build_kernel()
    with pytest.raises(RuntimeError):
        # Expecting an error due to type mismatch within the kernel launch.
        _ = kl_div_module.forward(predictions.contiguous(), targets.contiguous())

# Test 4: Trigger potential issue with the use of min in device code.
# We choose an input size such that n is not a multiple of blockDim.x (256).
def test_non_multiple_input_size():
    batch_size = 3
    dims = 1000  # 3000 elements total, not a multiple of 256.
    predictions = torch.rand(batch_size, dims, device="cuda", dtype=torch.float32)
    predictions = predictions.softmax(dim=-1)
    targets = torch.rand(batch_size, dims, device="cuda", dtype=torch.float32)
    targets = targets.softmax(dim=-1)

    log_predictions = torch.log(predictions)
    kl_div_module = build_kernel()
    
    # The kernel may read outside bounds or compute incorrectly if min is not handled correctly.
    custom_out = kl_div_module.forward(log_predictions.contiguous(), targets.contiguous())
    
    # Here we do not have an exact expected value but ensure the kernel runs without crashing.
    assert custom_out is not None, "Kernel did not return a valid output for non-multiple input size."

if __name__ == '__main__':
    pytest.main([__file__])
