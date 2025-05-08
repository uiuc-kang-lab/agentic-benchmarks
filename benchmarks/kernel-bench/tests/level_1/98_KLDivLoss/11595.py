
import torch
import pytest
from torch.nn.functional import kl_div as torch_kl_div
from torch.utils.cpp_extension import load
import math

# Utility to build the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# ---------------------------
# Test 1: Wrong formula and normalization factor
#
# For a simple case where predictions and targets are known,
# the PyTorch built-in kl_div returns a specific value according to
# loss = sum(target*(log(target)-log_pred)) / batch_size.
# The kernel, however, computes (exp(log_pred)- target*log_pred)/n.
#
# For example, if predictions = [0.5, 0.5] then log_predictions = [log(0.5), log(0.5)]
# and for targets = [0.5, 0.5] the expected loss (using batchmean reduction) is 0,
# while the kernel will compute a nonzero value.
def test_incorrect_formula_normalization():
    cuda_module = build_kernel()
    batch_size = 2
    feature_size = 2
    # Create a simple distribution so that predictions==targets==[0.5, 0.5] on each sample.
    # Note: both inputs need to be on CUDA and in float32.
    predictions = torch.full((batch_size, feature_size), 0.5, dtype=torch.float32, device='cuda')
    # Compute log_predictions manually
    log_predictions = predictions.log()
    targets = predictions.clone()
    
    # Compute built-in PyTorch KL divergence:
    # NOTE: torch.nn.functional.kl_div expects log_predictions and target probabilities.
    expected = torch_kl_div(log_predictions, targets, reduction='batchmean')
    
    # Compute kernel output:
    kernel_out = cuda_module.forward(log_predictions, targets)
    torch.cuda.synchronize()
    
    # The expected value should be 0, while the custom kernel does a different computation.
    assert not torch.allclose(kernel_out, expected, atol=1e-5), (
        f"Kernel output unexpectedly matches expected output. Kernel: {kernel_out.item()}, Expected: {expected.item()}"
    )

# ---------------------------
# Test 2: Non-contiguous input tensors
#
# The kernel directly accesses the data pointer assuming contiguous memory.
# This test creates non-contiguous tensors by transposing a tensor.
def test_non_contiguous_input():
    cuda_module = build_kernel()
    batch_size = 4
    feature_size = 8
    # Create contiguous tensor then create a non-contiguous view (e.g. transpose)
    a = torch.randn(batch_size, feature_size, dtype=torch.float32, device='cuda')
    b = torch.randn(batch_size, feature_size, dtype=torch.float32, device='cuda')
    # Make a non-contiguous version by unsqueezing and then transposing and finally squeezing back.
    log_predictions = a.unsqueeze(0).transpose(0, 1).squeeze(1).log()
    targets = b.unsqueeze(0).transpose(0, 1).squeeze(1).softmax(dim=-1)
    
    # Check if the tensor is non-contiguous:
    assert not log_predictions.is_contiguous(), "Test setup failure: log_predictions tensor should be non-contiguous."
    assert not targets.is_contiguous(), "Test setup failure: targets tensor should be non-contiguous."

    # The kernel does not handle non-contiguous memory: this might lead to incorrect results.
    kernel_out = cuda_module.forward(log_predictions, targets)
    torch.cuda.synchronize()
    
    # We compute reference using built-in function on contiguous tensors:
    log_predictions_contig = log_predictions.contiguous()
    targets_contig = targets.contiguous()
    expected = torch_kl_div(log_predictions_contig, targets_contig, reduction='batchmean')
    
    # The results should differ because the kernel incorrectly assumes contiguity.
    assert not torch.allclose(kernel_out, expected, atol=1e-4), (
        f"Kernel output ({kernel_out.item()}) appears correct with non-contiguous inputs, which is unexpected."
    )

# ---------------------------
# Test 3: Input tensor type (non float32)
#
# The kernel expects float32 inputs. If double inputs are passed in,
# the kernel will interpret the bits incorrectly. This test checks for
# a runtime error or an incorrect result.
def test_input_tensor_type():
    cuda_module = build_kernel()
    batch_size = 8
    feature_size = 16
    # Create double precision tensors on CUDA.
    predictions = torch.randn(batch_size, feature_size, dtype=torch.double, device='cuda').softmax(dim=-1)
    log_predictions = predictions.log()
    targets = predictions.clone()
    
    # The kernel expects float data. We try to run it with double tensors.
    with pytest.raises(RuntimeError):
        _ = cuda_module.forward(log_predictions, targets)

# ---------------------------
# Test 4: Non-power-of-two block dimension assumptions
#
# The reduction in shared memory assumes that the blockDim is a power of two.
# We trigger this issue by passing in an input whose total number of elements
# forces the kernel launch to use a non power-of-two number of threads per block
# (for example, by modifying the launch parameters via a input size).
def test_non_power_of_two_thread_count(monkeypatch):
    cuda_module = build_kernel()
    
    # We'll simulate a scenario by using an input size that may result in a kernel launch
    # with a non-power-of-two thread block if the user decided to use a block size
    # based on input dimensions. Since our kernel code hardcodes 256 threads,
    # we simulate a situation by using a small input size.
    batch_size = 3  # small batch size to force fewer elements than the thread block size
    feature_size = 5
    predictions = torch.randn(batch_size, feature_size, dtype=torch.float32, device='cuda').softmax(dim=-1)
    log_predictions = predictions.log()
    targets = predictions.clone()
    
    kernel_out = cuda_module.forward(log_predictions, targets)
    torch.cuda.synchronize()
    
    # Since the blockDim is 256 regardless, the reduction loop might over-read if n < blockDim.
    # We check that the output differs from what PyTorch calculates.
    expected = torch_kl_div(log_predictions, targets, reduction='batchmean')
    # The results should differ.
    assert not torch.allclose(kernel_out, expected, atol=1e-5), (
        f"Kernel output ({kernel_out.item()}) unexpectedly matches expected output with non-power-of-two input size."
    )
