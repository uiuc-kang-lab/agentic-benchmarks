
import pytest
import torch
from torch.utils.cpp_extension import load
import os

def build_kernel():
    # Build the extension from kernel.cu in the current directory.
    cuda_module = load(
        name="hinge_loss_kernel",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Pass non-float tensor (double type) to check that the kernel fails.
def test_non_float_dtype():
    my_kernel = build_kernel()
    # Create double precision tensors on CUDA.
    predictions = torch.randn(128, device="cuda", dtype=torch.double)
    targets = (torch.randint(0, 2, (128,), device="cuda", dtype=torch.double) * 2 - 1)
    with pytest.raises(RuntimeError):
        # Expect the kernel to either crash or throw an error due to type misinterpretation.
        my_kernel.forward(predictions, targets)
    torch.cuda.synchronize()

# Test 2: Pass non-contiguous tensor to trigger the CHECK_CONTIGUOUS macro.
def test_non_contiguous_input():
    my_kernel = build_kernel()
    predictions = torch.randn(128, device="cuda", dtype=torch.float32)
    targets = (torch.randint(0, 2, (128,), device="cuda", dtype=torch.float32) * 2 - 1)
    # Make predictions non-contiguous by transposing an expanded tensor.
    predictions = predictions.unsqueeze(1).transpose(0, 1).squeeze(1)
    assert not predictions.is_contiguous()
    with pytest.raises(RuntimeError):
        my_kernel.forward(predictions, targets)
    torch.cuda.synchronize()

# Test 3: Force a situation where the hardcoded block size and warp unrolling assumptions become problematic.
def test_small_input_n_less_than_required_warp_unroll():
    my_kernel = build_kernel()
    # Although the kernel launch always uses 256 threads per block,
    # we simulate a case with very few valid elements (n < 64). 
    # Many threads will index out of range of logical input, but will compute 0.
    # This tests that the reduction code may access shared_sum indices (e.g. tid+32)
    # even though only a few threads produced a nonzero result.
    n = 16
    predictions = torch.randn(n, device="cuda", dtype=torch.float32)
    # Targets in {-1, 1}
    targets = (torch.randint(0, 2, (n,), device="cuda", dtype=torch.float32) * 2 - 1)
    # This kernel invocation will launch at least one block with 256 threads.
    # Due to the assumptions in the kernel, the reduction code may be unsafe if blockDim < 64.
    # Here, the kernel is mis‐configured (it assumes blockDim==256) and the test should catch it by
    # either a runtime error or an incorrect result.
    loss = my_kernel.forward(predictions, targets)
    torch.cuda.synchronize()
    # Compute reference hinge loss on CPU.
    hinge = torch.clamp(1 - predictions * targets, min=0)
    ref_loss = torch.mean(hinge)
    # We expect a mismatch due to block reduction issues.
    assert not torch.allclose(loss, ref_loss, atol=1e-5), "Kernel result unexpectedly matches reference result."

# Test 4: Pass empty input to trigger potential division by zero.
def test_empty_input():
    my_kernel = build_kernel()
    predictions = torch.empty(0, device="cuda", dtype=torch.float32)
    targets = torch.empty(0, device="cuda", dtype=torch.float32)
    with pytest.raises(ZeroDivisionError):
        loss = my_kernel.forward(predictions, targets)
        # Force evaluation that might trigger division by zero.
        _ = loss.item()
    torch.cuda.synchronize()
