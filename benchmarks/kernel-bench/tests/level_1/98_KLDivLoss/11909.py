
import torch
import pytest
from torch.nn import functional as F
from torch.utils.cpp_extension import load

# Utility function: compile and load the kernel extension.
def build_kernel(extra_cuda_cflags=None):
    cuda_module = load(
        name="kl_div_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=extra_cuda_cflags if extra_cuda_cflags is not None else ["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1: Reduction factor is off.
# Test: For a given batch and input shape, the loss computed by our kernel
# (which divides by n) will differ from that computed by torch.nn.functional.kl_div.
def test_reduction_factor_issue():
    # create a test case where batch_size is not equal to total elems (n)
    batch_size = 4
    num_elems = 16  # e.g. shape = (batch_size, num_elems) so n = 64.
    # Create valid probabilities with softmax
    predictions = torch.randn(batch_size, num_elems, device="cuda").softmax(dim=-1)
    # Must be valid probability distribution (non-zero)
    targets = torch.randn(batch_size, num_elems, device="cuda").softmax(dim=-1)
    # Our CUDA kernel expects inputs: log(predictions) and targets.
    log_predictions = torch.log(predictions)
    my_module = build_kernel()
    # Kernel output
    kernel_out = my_module.forward(log_predictions, targets)
    # PyTorch reference uses reduction='batchmean' -> division by batch_size.
    ref = F.kl_div(log_predictions, targets, reduction="batchmean")
    # Because our kernel divides by n and not by batch_size,
    # the result will be off by a factor equal to (batch_size/n_elements)
    # So we expect the two results to be different.
    assert not torch.allclose(kernel_out, ref, atol=1e-5), (
        f"Reduction factor issue not triggered: kernel_out={kernel_out.item()}, ref={ref.item()}"
    )

# Issue 2: Incorrect arithmetic for KL divergence.
# Test: For a given input, the kernel computation (predictions - targets*log_predictions)
# does not match the standard KL divergence (i.e. targets*(log(targets)-log_predictions)).
def test_incorrect_kl_div_arithmetic():
    # Use a simple case where we have nontrivial differences.
    batch_size = 2
    num_elems = 8
    predictions = torch.randn(batch_size, num_elems, device="cuda").softmax(dim=-1)
    targets = torch.randn(batch_size, num_elems, device="cuda").softmax(dim=-1)
    log_predictions = torch.log(predictions)
    my_module = build_kernel()
    kernel_out = my_module.forward(log_predictions, targets)
    
    # Compute what the kernel is computing manually:
    # It does sum( predictions - targets*log_predictions)/n .
    n = predictions.numel()
    manual = (predictions - targets * log_predictions).sum() / n

    # Compute correct KL divergence per PyTorch's formula (ignoring the constant term that does not depend on predictions)
    # PyTorch defines: kl_div = sum( targets * (log(targets)-log_predictions) ) / batch_size
    correct = (targets * (torch.log(targets) - log_predictions)).sum() / batch_size

    # Since both the arithmetic and the division factor are different, the kernel result should not match the "correct" value.
    assert not torch.allclose(kernel_out, correct, atol=1e-5), (
        f"Kernel arithmetic issue not triggered: kernel_out={kernel_out.item()}, correct={correct.item()}"
    )

# Issue 3: Hard-coded block size (and derived shared memory allocation).
# Test: Build the kernel with a different block size (via a macro override)
# and check that the final result (which uses BLOCK_SIZE and shared memory array size)
# now produces an inconsistent result relative to torch.nn.functional.kl_div.
def test_block_size_incompatibility():
    # Override BLOCK_SIZE to a non-standard value
    extra_flags = ["-O3", "--use_fast_math", "-DBLOCK_SIZE=128"]
    batch_size = 4
    num_elems = 16
    predictions = torch.randn(batch_size, num_elems, device="cuda").softmax(dim=-1)
    targets = torch.randn(batch_size, num_elems, device="cuda").softmax(dim=-1)
    log_predictions = torch.log(predictions)
    my_module = build_kernel(extra_cuda_cflags=extra_flags)
    kernel_out = my_module.forward(log_predictions, targets)
    ref = F.kl_div(log_predictions, targets, reduction="batchmean")
    # Because of the block size assumption embedded in the kernel,
    # using a different BLOCK_SIZE (here 128 instead of 256) should trigger an error in reduction.
    assert not torch.allclose(kernel_out, ref, atol=1e-5), (
        f"Block size compatibility issue not triggered: kernel_out={kernel_out.item()}, ref={ref.item()}"
    )

# Issue 4: Lack of type checking (e.g. using float64 inputs).
# Test: Pass input tensors in float64 and expect the kernel to raise an error or produce an incorrect result.
def test_input_tensor_type():
    batch_size = 4
    num_elems = 16
    # Create tensors with double precision.
    predictions = torch.randn(batch_size, num_elems, device="cuda", dtype=torch.float64).softmax(dim=-1)
    targets = torch.randn(batch_size, num_elems, device="cuda", dtype=torch.float64).softmax(dim=-1)
    # The kernel expects float32, so we expect an error (or an incorrect result)
    my_module = build_kernel()
    log_predictions = torch.log(predictions)
    with pytest.raises(RuntimeError):
        # Most likely the kernel call (or atomicAdd on the wrong type) will trigger an error.
        _ = my_module.forward(log_predictions, targets)
