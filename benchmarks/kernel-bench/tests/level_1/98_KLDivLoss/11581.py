
import pytest
import torch
from torch.utils.cpp_extension import load
import math

def build_kernel(extra_cuda_cflags=None):
    args = {
        "name": "kl_div_ext",
        "sources": ["kernel.cu"],
        "with_cuda": True,
        "verbose": False,
    }
    if extra_cuda_cflags is not None:
        args["extra_cuda_cflags"] = extra_cuda_cflags
    return load(**args)

# -------------------------------
# Test 1: Input tensor type issue.
# This test checks that if the input tensors are not float32, the kernel either fails or
# returns a result that is inconsistent with reference behavior.
def test_input_tensor_type():
    my_module = build_kernel()
    # Create double (float64) tensors on CUDA.
    # Since the kernel only supports float, we expect an error or a wrong result.
    batch_size = 10
    input_shape = (100,)
    # Use a double type input even though we call softmax (later casts internally but here kernel expects float32)
    predictions = torch.randn(batch_size, *input_shape, device='cuda', dtype=torch.double).softmax(dim=-1)
    targets = torch.randn(batch_size, *input_shape, device='cuda', dtype=torch.double).softmax(dim=-1)
    with pytest.raises(Exception):
        # We expect a failure (or at least an incorrect usage) when non-float32 inputs are provided.
        res = my_module.forward(predictions, targets)
        torch.cuda.synchronize()

# -------------------------------
# Test 2: Normalization mismatch.
# The kernel computes the divergence sum over all elements and then divides by n rather than by batch_size.
def test_normalization():
    my_module = build_kernel()
    # Choose an input shape similar to the provided model example.
    batch_size = 128
    n_features = 4096
    # Create two tensors in float32 (as required)
    # (We use softmax so that predictions are probabilities and log is valid)
    predictions = torch.randn(batch_size, n_features, device='cuda', dtype=torch.float32).softmax(dim=-1)
    targets = torch.randn(batch_size, n_features, device='cuda', dtype=torch.float32).softmax(dim=-1)
    # Run the kernel
    divergence = my_module.forward(predictions, targets)
    torch.cuda.synchronize()
    
    # Compute the divergence using the same formula as in the kernel:
    # f(log_pred, target) = exp( log_pred ) - target * log_pred
    # (Note: the normal kl_div in torch divides by batch_size when reduction='batchmean')
    log_preds = torch.log(predictions)
    divergence_elements = torch.exp(log_preds) - targets * log_preds
    sum_div = divergence_elements.sum()
    # Kernel divides by total number of elements instead of batch size.
    expected_kernel_norm = sum_div / (batch_size * n_features)
    # The correct "batchmean" normalization should divide by batch_size.
    expected_batchmean = sum_div / batch_size

    # Check that the kernel output does not match the correct batchmean normalization.
    # (In a correct kernel implementation the following condition should hold:
    #  divergence ≈ expected_batchmean.
    #  Here we expect divergence ≈ expected_kernel_norm.)
    rel_diff = abs(divergence.item() - expected_batchmean.item()) / abs(expected_batchmean.item())
    assert rel_diff > 0.1, f"Normalization error: kernel output {divergence.item()} is too close to correct batchmean {expected_batchmean.item()} (relative diff={rel_diff:.3f})"


# -------------------------------
# Test 3: Block reduction assumptions (non-power-of-two thread block)
# The kernel hardcodes blockDim.x = 256 and implements the reduction assuming a power-of-two.
# In more general cases (e.g. if one later wants to launch with a non-power-of-two block size)
# the reduction code may behave incorrectly. Here we simulate this by re-compiling the kernel
# with a custom macro that changes the thread block size.
#
# (Note: In the supplied source the thread count is hard-coded to 256.
#  For testing purposes, we assume that later the developer may want to compile with a different
#  thread count by defining a macro. This test will fail if the reduction cannot handle a non-power-of-two block size.)
def test_non_power_two_block_dimension():
    # Rebuild the kernel with a compile-time define to change the number of threads.
    # Suppose we add the -DTHREADS=33 flag and the kernel code internally uses:
    #   const int threads = #ifdef THREADS ... #else 256 #endif
    # Even though our given kernel does not do that, this test serves as a reminder that if support for
    # non-power-of-two thread block sizes is added, the reduction code must be validated.
    my_module = build_kernel(extra_cuda_cflags=["-DTHREADS=33"])
    
    # Prepare a tensor with a number of elements that forces one block launch.
    N = 33  # non-power-of-two count of total elements
    predictions = torch.randn(N, device='cuda', dtype=torch.float32).softmax(dim=0)
    targets = torch.randn(N, device='cuda', dtype=torch.float32).softmax(dim=0)
    divergence = my_module.forward(predictions, targets)
    torch.cuda.synchronize()

    # Compute expected result (note: the kernel divides by N, as in the current implementation)
    log_preds = torch.log(predictions)
    divergence_elements = torch.exp(log_preds) - targets * log_preds
    expected = divergence_elements.sum() / N

    # If the block reduction code were not written to handle non-power-of-two thread counts correctly,
    # the sum would be off.
    assert torch.allclose(divergence, expected, atol=1e-5), \
           f"Block reduction issue: kernel output {divergence.item()} does not match expected {expected.item()}"

