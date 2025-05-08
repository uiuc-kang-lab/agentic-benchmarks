
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Helper to run the kernel-based KL divergence forward
def run_kernel(log_predictions, targets, module):
    # The kernel returns a tensor with a single element.
    out = module.forward(log_predictions, targets)
    # Synchronize to ensure kernel has finished.
    torch.cuda.synchronize()
    return out

# Issue 1: Test that using input tensors that are not float32 (e.g. double) leads to incorrect behavior.
def test_wrong_tensor_dtype():
    # Create double precision tensors on CUDA.
    batch_size = 128
    features = 4096
    # Using softmax to mimic the model input.
    predictions = torch.randn(batch_size, features, device="cuda", dtype=torch.float64)
    predictions = torch.softmax(predictions, dim=-1)
    targets = torch.randn(batch_size, features, device="cuda", dtype=torch.float64)
    targets = torch.softmax(targets, dim=-1)
    
    module = build_kernel()
    # This kernel implementation always casts data_ptr to float*.
    # As a result the computations will be wrong.
    out = run_kernel(predictions, targets, module)
    
    # Compute PyTorch reference using the built-in function (which will work fine with double)
    ref = F.kl_div(torch.log(predictions), targets, reduction="batchmean")
    # They should differ by more than a small epsilon because of type mismatch.
    assert not torch.allclose(out, ref, atol=1e-4), (
        f"Kernel output appears to be correct with wrong dtype. Got {out.item()} vs ref {ref.item()}"
    )

# Issue 2 & 3: Test that the kernel implements a wrong formula and uses an incorrect reduction factor.
def test_incorrect_formula_and_reduction():
    batch_size = 128
    features = 4096
    # Create well-behaved float32 tensors.
    predictions = torch.randn(batch_size, features, device="cuda", dtype=torch.float32)
    predictions = torch.softmax(predictions, dim=-1)
    targets = torch.randn(batch_size, features, device="cuda", dtype=torch.float32)
    targets = torch.softmax(targets, dim=-1)
    
    module = build_kernel()
    out = run_kernel(predictions.log(), targets, module)
    # Expected result using PyTorch’s built-in F.kl_div.
    expected = F.kl_div(torch.log(predictions), targets, reduction="batchmean")
    # Because the kernel does:
    #    sum(predictions - targets * log_predictions) / (batch_size*features)
    # instead of the proper sum(targets*(log(targets)-log_predictions)) / batch_size,
    # the result will be off both in absolute value and scaling.
    # We check that the relative error is significant.
    rel_error = abs(out.item() - expected.item()) / abs(expected.item() + 1e-8)
    assert rel_error > 0.01, (
        f"Kernel output seems to match expected value. Relative error: {rel_error:.4f}. "
        f"This indicates a missing term or wrong reduction factor."
    )

# Issue 1 (alignment): Test that passing non-contiguous tensors (which might not be properly aligned)
# leads to a divergence from the expected behavior.
def test_non_contiguous_tensors():
    batch_size = 128
    features = 4096
    A = torch.randn(batch_size, features + 1, device="cuda", dtype=torch.float32)
    # Slice a non-contiguous view (skip the first column) to get a tensor that might violate the
    # alignment assumptions for vectorized loads.
    predictions = A[:, 1:]
    predictions = torch.softmax(predictions, dim=-1)
    B = torch.randn(batch_size, features + 1, device="cuda", dtype=torch.float32)
    targets = B[:, 1:]
    targets = torch.softmax(targets, dim=-1)
    
    # Make sure the tensors are non-contiguous.
    assert not predictions.is_contiguous(), "Predictions tensor is contiguous but should be non-contiguous for this test."
    module = build_kernel()
    out = run_kernel(predictions.log(), targets, module)
    # Compute the expected result from PyTorch's F.kl_div.
    expected = F.kl_div(torch.log(predictions), targets, reduction="batchmean")
    # With non-contiguous misaligned inputs the kernel’s vectorized load might cause errors or wrong results.
    rel_error = abs(out.item() - expected.item()) / abs(expected.item() + 1e-8)
    assert rel_error > 0.01, (
        f"Kernel output appears to be correct with non-contiguous inputs. Relative error: {rel_error:.4f}, "
        "which is unexpected given alignment assumptions."
    )
    
if __name__ == "__main__":
    pytest.main([__file__])
