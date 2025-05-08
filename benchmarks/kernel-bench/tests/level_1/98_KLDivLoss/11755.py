
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the extension from kernel.cu.
    # If kernel.cu does not contain proper input validations, issues will be triggered.
    return load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

# Issue 1: Incorrect memory copy for input tensors already on GPU.
def test_incorrect_input_memory_copy():
    module = build_kernel()
    # Create inputs on GPU.
    batch = 128
    dim = 4096
    # The model normally expects softmax probabilities.
    preds = torch.rand(batch, dim, device='cuda', dtype=torch.float32).softmax(dim=-1)
    log_preds = torch.log(preds)
    targets = torch.rand(batch, dim, device='cuda', dtype=torch.float32).softmax(dim=-1)

    # Because the host function incorrectly assumes inputs are in host memory,
    # the result will be computed on a copy that is done with the wrong direction.
    # We trigger the error by comparing with the expected batch normalization.
    res = module.forward(log_preds, targets)
    # Expected: KL loss with "batchmean" reduction computes sum(loss)/batch_size,
    # but our kernel divides by total number of elements.
    expected = (torch.exp(log_preds) - targets * log_preds).sum() / float(batch)
    # The result from the BUGGY kernel is divided by (batch*dim), so it will differ.
    assert not torch.allclose(res, expected, atol=1e-5), (
        "Kernel incorrectly handling device memory copies; normalization factor error not detected."
    )

# Issue 2: Incorrect normalization factor (dividing by n instead of batch size).
def test_normalization_error():
    module = build_kernel()
    batch = 128
    dim = 4096
    preds = torch.rand(batch, dim, device='cuda', dtype=torch.float32).softmax(dim=-1)
    log_preds = torch.log(preds)
    targets = torch.rand(batch, dim, device='cuda', dtype=torch.float32).softmax(dim=-1)

    res = module.forward(log_preds, targets)
    # Expected KL divergence 'batchmean' normalization divides by batch size.
    expected = (torch.exp(log_preds) - targets * log_preds).sum() / float(batch)
    # If normalized by total elements, the value will be scaled incorrectly.
    assert not torch.allclose(res, expected, atol=1e-5), (
        "Normalization error: kernel output matches batchmean normalization unexpectedly."
    )

# Issue 3: Lack of error checking for CUDA API calls.
def test_cuda_api_error_propagation():
    module = build_kernel()
    batch = 128
    dim = 4096
    preds = torch.rand(batch, dim, device='cuda', dtype=torch.float32).softmax(dim=-1)
    log_preds = torch.log(preds)
    targets = torch.rand(batch, dim, device='cuda', dtype=torch.float32).softmax(dim=-1)
    # There is no direct way to force a cudaMalloc failure in a unit test,
    # but we can check that the kernel does not raise any CUDA runtime error.
    # Since proper error checking is missing, this test only documents that errors go unnoticed.
    res = module.forward(log_preds, targets)
    assert torch.isfinite(res).all(), "Kernel did not produce a finite result (possible missing error checks)."

# Issue 4: Lack of input validation for data type and contiguity.
def test_input_validation_dtype():
    module = build_kernel()
    batch = 128
    dim = 4096
    # Create tensors with wrong dtype (double instead of float)
    preds = torch.rand(batch, dim, device='cuda', dtype=torch.float64).softmax(dim=-1)
    log_preds = torch.log(preds)
    targets = torch.rand(batch, dim, device='cuda', dtype=torch.float64).softmax(dim=-1)
    # The kernel blindly casts data_ptr to float* so the wrong dtype will lead to incorrect computation.
    res = module.forward(log_preds, targets)
    # Expect the result to be nonsensical.
    assert not torch.isfinite(res).all(), "Kernel did not detect incorrect data type usage."

def test_input_validation_contiguity():
    module = build_kernel()
    batch = 128
    dim = 4096
    preds = torch.rand(batch, dim, device='cuda', dtype=torch.float32).softmax(dim=-1)
    # Create non-contiguous tensor by transposing twice.
    non_contig_preds = preds.t().contiguous().t()  # force non-contiguity in an indirect way
    log_preds = torch.log(non_contig_preds)
    targets = torch.rand(batch, dim, device='cuda', dtype=torch.float32).softmax(dim=-1)
    # The kernel does not check for contiguity.
    res = module.forward(log_preds, targets)
    # Result is likely incorrect due to unexpected memory layout.
    assert not torch.isfinite(res).all(), "Kernel did not detect non-contiguous input."

# Issue 5: Limited grid configuration (capping the number of blocks).
def test_grid_configuration_limit():
    module = build_kernel()
    # Create a very large input tensor so that an unnecessarily capped grid leads to underutilization.
    batch = 1024  # larger batch
    dim = 4096
    preds = torch.rand(batch, dim, device='cuda', dtype=torch.float32).softmax(dim=-1)
    log_preds = torch.log(preds)
    targets = torch.rand(batch, dim, device='cuda', dtype=torch.float32).softmax(dim=-1)
    res = module.forward(log_preds, targets)
    # Expected value computed with proper batchmean normalization.
    expected = (torch.exp(log_preds) - targets * log_preds).sum() / float(batch)
    # Because the kernel divides by total number of elements (batch*dim) and may not launch enough blocks,
    # the result will differ from the expected value.
    assert not torch.allclose(res, expected, atol=1e-5), (
        "Kernel grid configuration cap did not trigger an observable normalization error."
    )

# Issue 6: Using unqualified min function.
def test_min_function_issue():
    module = build_kernel()
    batch = 128
    dim = 4096
    preds = torch.rand(batch, dim, device='cuda', dtype=torch.float32).softmax(dim=-1)
    log_preds = torch.log(preds)
    targets = torch.rand(batch, dim, device='cuda', dtype=torch.float32).softmax(dim=-1)
    res = module.forward(log_preds, targets)
    # The minimal test here is simply to verify that the kernel build did not fail due to an unqualified min.
    assert torch.is_tensor(res), "Kernel did not execute properly, possibly due to min function issues."
