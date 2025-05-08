
import pytest
import torch
from torch.utils.cpp_extension import load

# Utility function to compile and load the CUDA kernel from kernel.cu.
def build_kernel():
    # Note: This assumes that kernel.cu is in the same directory.
    return load(
        name="kl_div_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

# Issue 1 Test: Incorrect KL divergence formula
def test_incorrect_formula():
    # Create a simple two-element batch so we can compute the expected loss.
    # For a uniform distribution, expected KL divergence (with predictions==targets) is 0.
    batch_size = 2
    n = 2
    # Use uniform predictions
    pred = torch.full((batch_size, n), 1.0 / n, device="cuda", dtype=torch.float32)
    # The kernel expects log_predictions (i.e. log(prob))
    log_pred = pred.log()
    # And targets, which is the same in this test case.
    target = pred.clone()
    
    kl_mod = build_kernel()
    # Invoke the CUDA kernel via the bound forward function.
    out = kl_mod.forward(log_pred.contiguous(), target.contiguous())
    torch.cuda.synchronize()
    
    # Expected KL divergence for identical distributions should be 0.
    # Due to the incorrect formula, the kernel does not return 0.
    # We check that the output differs from 0 by a margin.
    assert not torch.allclose(out, torch.zeros_like(out), atol=1e-6), \
        f"Expected non-zero output because of incorrect formula, but got {out.item()}"

# Issue 2 Test: Incorrect normalization factor
def test_incorrect_normalization():
    # This test is designed to show that the normalization factor (dividing by n)
    # does not match the batchmean division (which should divide by batch size).
    batch_size = 4
    n = 8  # number of classes
    # Create simple tensors with non-uniform but known probabilities.
    # We set up predictions and targets such that the proper KL divergence (batchmean)
    # can be calculated in Python.
    pred = torch.rand(batch_size, n, device="cuda", dtype=torch.float32)
    pred = pred.softmax(dim=-1)
    log_pred = pred.log()
    target = torch.rand(batch_size, n, device="cuda", dtype=torch.float32)
    target = target.softmax(dim=-1)
    
    # Compute reference using torch's native kl_div: note that kl_div uses reduction='batchmean'
    ref = torch.nn.functional.kl_div(log_pred, target, reduction='batchmean')
    
    kl_mod = build_kernel()
    out = kl_mod.forward(log_pred.contiguous(), target.contiguous())
    torch.cuda.synchronize()
    
    # According to kernel implementation, the output is divided by total number of elements,
    # which is (batch_size * n) instead of batch_size.
    # They will differ unless by chance n == 1.
    # We expect a significant difference.
    diff = abs(out.item() - ref.item())
    assert diff > 1e-3, f"Kernel normalization error: expected difference in normalization, but got diff={diff}"

# Issue 3 Test: Wrong dtype (double instead of float32)
def test_wrong_dtype():
    # Create tensors of double type.
    batch_size = 2
    n = 4
    pred = torch.rand(batch_size, n, device="cuda", dtype=torch.float64)
    pred = pred.softmax(dim=-1)
    log_pred = pred.log()
    target = pred.clone()
    
    kl_mod = build_kernel()
    # Expect the kernel to misbehave or throw an error when given double precision buffers.
    with pytest.raises(RuntimeError):
        # This call should fail because the kernel uses data_ptr<float>()
        _ = kl_mod.forward(log_pred.contiguous(), target.contiguous())
    torch.cuda.synchronize()

# Issue 4 Test: Non‐contiguous inputs
def test_non_contiguous_input():
    # Create contiguous tensors and then create non-contiguous views.
    batch_size = 4
    n = 8
    pred = torch.rand(batch_size, n, device="cuda", dtype=torch.float32)
    pred = pred.softmax(dim=-1)
    log_pred = pred.log()
    target = pred.clone()
    
    # Create a non-contiguous view by transposing (if possible)
    log_pred_nc = log_pred.t()  # now shape is (n, batch_size) and non-contiguous
    target_nc = target.t()
    
    kl_mod = build_kernel()
    # The kernel expects contiguous memory. Passing non-contiguous tensors should lead
    # to incorrect behavior. We expect the test to fail the correctness check.
    out = kl_mod.forward(log_pred_nc, target_nc)
    torch.cuda.synchronize()

    # Compute expected reference using PyTorch's kernel on contiguous tensors:
    ref = torch.nn.functional.kl_div(log_pred, target, reduction='batchmean')
    # Since non-contiguous tensors might lead to wrong results, assert that they are not close.
    assert not torch.allclose(out, ref, atol=1e-4), \
        f"Kernel did not fail on non-contiguous input as expected, output {out.item()} is too close to reference {ref.item()}"
