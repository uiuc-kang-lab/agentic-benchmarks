
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build and load the CUDA extension from kernel.cu
def build_kernel():
    # Note: Using a temporary directory to compile the extension if needed.
    cuda_module = load(
        name="kl_div_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test case 1: Trigger issue with input tensor data type (double instead of float)
def test_wrong_dtype():
    my_module = build_kernel()
    N = 4096
    # Create inputs in double precision; kernel expects float32.
    pred = torch.randn(128, N, dtype=torch.float64, device="cuda").softmax(dim=-1)
    target = torch.randn(128, N, dtype=torch.float64, device="cuda").softmax(dim=-1)
    with pytest.raises(RuntimeError):
        # Expect the kernel launch or pointer cast to fail due to type mismatch.
        _ = my_module.forward(pred, target)
    torch.cuda.synchronize()

# Test case 2: Check for incorrect mathematical formula relative to correct kl_div (omitted target*log(target))
def test_incorrect_formula():
    my_module = build_kernel()
    # Create small but non-uniform distributions
    batch_size, n = 4, 10
    # Use softmax over last dimension to get probabilities.
    pred = torch.randn(batch_size, n, device="cuda", dtype=torch.float32).softmax(dim=-1)
    # The forward input is log(predictions), so compute it.
    log_pred = torch.log(pred)
    # Use random positive targets that sum to one per sample
    target = torch.randn(batch_size, n, device="cuda", dtype=torch.float32).softmax(dim=-1)
    
    # Kernel forward
    kernel_out = my_module.forward(log_pred, target)
    torch.cuda.synchronize()
    
    # Compute correct kl_div according to PyTorch definition:
    # kl_div = sum(target * (log(target) - log(prediction))) / batch_size
    target_log = torch.log(target + 1e-10)  # Add small epsilon for stability
    correct_kl = torch.sum(target * (target_log - log_pred)) / batch_size

    # Since the kernel computes: sum(exp(log_pred) - target * log_pred) / total_elements,
    # the result will differ both in missing target*log(target) and in the normalization factor.
    # We trigger the issue by checking that the error exceeds a small tolerance.
    assert not torch.allclose(kernel_out, correct_kl, atol=1e-5), \
        f"Kernel output unexpectedly matches PyTorch kl_div. Kernel: {kernel_out.item()}, PyTorch: {correct_kl.item()}"

# Test case 3: Check for normalization issue (dividing by total num elements instead of batch size)
def test_normalization():
    my_module = build_kernel()
    # Create a case where each distribution is different in size.
    batch_size, n = 8, 256  # Here n_total = 8*256 = 2048, but reduction='batchmean' expects division by 8.
    pred = torch.randn(batch_size, n, device="cuda", dtype=torch.float32).softmax(dim=-1)
    log_pred = torch.log(pred)
    target = torch.randn(batch_size, n, device="cuda", dtype=torch.float32).softmax(dim=-1)
    
    # Kernel output uses division by (batch_size * n)
    kernel_out = my_module.forward(log_pred, target)
    torch.cuda.synchronize()
    
    # Correct kl_div using PyTorch built-in function
    correct_py_kl = torch.nn.functional.kl_div(log_pred, target, reduction='batchmean')
    
    # They should differ by roughly a factor of n (if the computed sum is similar) 
    # because kernel divides by (n*batch_size) while PyTorch divides by batch_size.
    # We trigger the issue by asserting that the normalized results are NOT close.
    assert not torch.allclose(kernel_out, correct_py_kl, atol=1e-5), \
        f"Normalization issue not detected: kernel {kernel_out.item()}, PyTorch {correct_py_kl.item()}"
