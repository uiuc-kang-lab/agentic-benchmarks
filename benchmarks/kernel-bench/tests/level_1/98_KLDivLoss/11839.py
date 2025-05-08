
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    # Compile the CUDA extension from kernel.cu
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 4: Test that using a tensor of incorrect type (e.g. double) triggers an error.
def test_input_tensor_type_error():
    cuda_module = build_kernel()
    N = 1024
    # Create double precision inputs, which the kernel is not designed for.
    A = torch.randn(N, device="cuda", dtype=torch.double)
    B = torch.randn(N, device="cuda", dtype=torch.double)
    with pytest.raises(RuntimeError):
        _ = cuda_module.forward(A, B)
    
# Issue 1: Test for synchronization issues.
#
# Because unsynchronized conditional barriers may yield nondeterministic behavior or hangs,
# we design a test that launches the kernel on a sufficiently large input (forcing many blocks)
# and then check that the output is a finite number. 
# (A hang will cause the test to timeout.)
def test_sync_issue():
    cuda_module = build_kernel()
    # Create an input whose total number of elements is large enough to require many blocks
    N = 1 << 20  # over a million elements
    # Use random values that are not pathological (but differences due to sync bug may occur)
    A = torch.log(torch.randn(N, device="cuda", dtype=torch.float32).abs() + 1.0)
    B = torch.rand(N, device="cuda", dtype=torch.float32)  # targets not normalized here
    out = cuda_module.forward(A, B)
    # Check that output is a finite real number.
    assert torch.isfinite(out).all(), "Output contains non-finite numbers; possible sync issue."

# Issue 2 & 3: Test that the formula and reduction factor are incorrect.
#
# For comparison, consider the case when predictions and targets are uniform.
# With uniform distributions and after applying softmax, torch.nn.functional.kl_div (with reduction 'batchmean')
# returns zero (since target * (log(target) - log_prediction) is zero for uniform distributions).
# However, our kernel computes, per element, exp(log_prediction) - target*log_prediction,
# which for uniform inputs does not vanish and further is averaged over the total number of elements.
def test_incorrect_division_and_formula():
    cuda_module = build_kernel()
    batch_size = 8
    feature_dim = 4096
    # Create uniform predictions and targets.
    # Note: predictions are softmax outputs so that torch.log(predictions) is the proper input.
    uniform_val = 1.0 / feature_dim
    predictions = torch.full((batch_size, feature_dim), uniform_val, device="cuda", dtype=torch.float32)
    log_predictions = torch.log(predictions)  # as expected by the CUDA kernel
    targets = torch.full((batch_size, feature_dim), uniform_val, device="cuda", dtype=torch.float32)
    
    # Run the CUDA kernel
    kernel_out = cuda_module.forward(log_predictions, targets)
    # Compute the reference KL divergence using PyTorch's functional API.
    # Using reduction 'batchmean' with uniform distributions yields 0.
    ref_out = torch.nn.functional.kl_div(log_predictions, targets, reduction='batchmean')
    # Because the kernel computes a nonzero value (due to both the wrong formula and dividing by n),
    # the two outputs will differ.
    # Here, we check that the kernel output is not close to the expected 0 value.
    assert not torch.allclose(kernel_out, ref_out, atol=1e-5), (
        f"Kernel output ({kernel_out.item()}) should not equal the reference output ({ref_out.item()})."
    )

if __name__ == '__main__':
    pytest.main([__file__])
