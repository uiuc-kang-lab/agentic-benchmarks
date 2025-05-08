
import pytest
import torch
from torch.utils.cpp_extension import load
import math

def build_kernel():
    cuda_module = load(
        name="kl_div_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Streaming branch condition is inverted.
# This test feeds huge CPU tensors so that the streaming branch is used, and then compares
# with the expected (PyTorch) result. Due to the faulty branch, we expect a mismatch.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_streaming_branch_inversion():
    # Create a huge tensor on CPU (n >= MIN_ELEMENTS_FOR_STREAMING)
    # MIN_ELEMENTS_FOR_STREAMING = 1<<22 = 4194304 elements; we create slightly above that.
    n_elems = (1 << 22) + 1000
    # Make a flat 1D tensor then reshape to 2D (batch, features)
    batch = 1
    feat = n_elems
    # Use softmax to make valid probability distributions.
    preds = torch.randn(batch, feat).softmax(dim=-1)
    targets = torch.randn(batch, feat).softmax(dim=-1)
    # They are on CPU so streaming branch will be used (which is intended for special CPU-to-D2H copy)
    kernel = build_kernel()
    # Call the kernel (which allocates output on CUDA even from CPU input)
    result = kernel.forward(preds, targets)
    # Get a reference using PyTorch's built-in function
    ref = torch.nn.functional.kl_div(torch.log(preds.cuda()), targets.cuda(), reduction='batchmean')
    # Because the streaming branch is not correct, we expect the numbers to differ.
    assert not torch.allclose(result, ref, atol=1e-4), \
         "Streaming branch executed correctly - but it was expected to be faulty due to inverted condition"

# Issue 2: Incorrect KL divergence computation.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_incorrect_kl_computation():
    # Use a small example to easily compute expected value.
    # For a distribution x with prediction and target, the correct kl_div (with reduction='batchmean')
    # is: sum(target * (log(target) - log(prediction))) / n.
    # The kernel computes: sum(exp(log_pred) - target*log_pred) / n.
    batch = 2
    feat = 8
    # Make sure our distributions are valid and reproducible
    torch.manual_seed(0)
    preds = torch.randn(batch, feat, device="cuda").softmax(dim=-1)
    targets = torch.randn(batch, feat, device="cuda").softmax(dim=-1)
    kernel = build_kernel()
    result = kernel.forward(torch.log(preds), targets)
    # Compute the expected kl divergence (as implemented in PyTorch)
    expected = torch.nn.functional.kl_div(torch.log(preds), targets, reduction='batchmean')
    
    # Since the kernel leaves out target*log(target) term, the results should differ.
    assert not torch.allclose(result, expected, atol=1e-4), \
         f"Kernel computed KL divergence correctly, but it was expected to be wrong. result: {result.item()}, expected: {expected.item()}"

# Issue 3: Non-contiguous input tensors are not supported.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_non_contiguous_inputs():
    batch = 4
    feat = 1024
    preds = torch.randn(batch, feat, device="cuda").softmax(dim=-1)
    targets = torch.randn(batch, feat, device="cuda").softmax(dim=-1)
    # Create non-contiguous views.
    preds_nc = preds.transpose(0, 1)
    targets_nc = targets.transpose(0, 1)
    kernel = build_kernel()
    # The kernel uses data_ptr assuming contiguous memory so the result will be incorrect.
    result = kernel.forward(torch.log(preds_nc), targets_nc)
    ref = torch.nn.functional.kl_div(torch.log(preds_nc.contiguous()), targets_nc.contiguous(), reduction='batchmean')
    assert not torch.allclose(result, ref, atol=1e-4), \
         "Kernel handled non-contiguous inputs correctly, but it was expected to fail given the lack of support."

# Issue 4: Kernel only supports float32. Passing double precision input should cause an error or produce wrong results.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_incorrect_input_dtype():
    batch = 4
    feat = 1024
    # Create double precision tensors
    preds = torch.randn(batch, feat, device="cuda", dtype=torch.double).softmax(dim=-1)
    targets = torch.randn(batch, feat, device="cuda", dtype=torch.double).softmax(dim=-1)
    kernel = build_kernel()
    with pytest.raises(RuntimeError):
        # Expecting an error because the kernel uses data_ptr<float>()
        kernel.forward(torch.log(preds), targets)

# Issue 5: Lack of error checking for CUDA API calls.
# We simulate a failure in CUDA allocation by attempting to launch the kernel on an extremely huge tensor.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_allocation_failure():
    # We attempt to allocate a tensor so huge that cudaMallocAsync is very likely to fail.
    # WARNING: This test intentionally tries to provoke a CUDA allocation error.
    huge_size = (1 << 30)  # 1 billion elements (approx 4GB for float32)
    # Create CPU inputs so that streaming branch is used.
    preds = torch.randn(1, huge_size).softmax(dim=-1)
    targets = torch.randn(1, huge_size).softmax(dim=-1)
    kernel = build_kernel()
    with pytest.raises(Exception):
        # Expecting an exception due to allocation failure or similar error.
        kernel.forward(torch.log(preds), targets)
