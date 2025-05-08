
import pytest
import torch
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

@pytest.fixture(scope="module")
def kernel_module():
    return build_kernel()

def test_tensor_type_issue(kernel_module):
    # Issue 1: Pass double tensors even though the kernel expects float32.
    # The kernel will interpret the underlying memory incorrectly, likely causing incorrect results or a runtime error.
    predictions = torch.randn(128, device="cuda", dtype=torch.double)
    # Generate tensor with values in {-1,1}
    targets = (torch.randint(0, 2, (128,), device="cuda", dtype=torch.double) * 2 - 1)
    with pytest.raises(RuntimeError):
        # When synchronizing, a CUDA error may be raised.
        out = kernel_module.forward(predictions, targets)
        # Force synchronization to trigger any latent error.
        torch.cuda.synchronize()

def test_min_usage_issue(kernel_module):
    # Issue 2: The improper usage of min() (without std::min or inclusion of <algorithm>)
    # typically causes a compilation error. However, if the module was built in an environment
    # where an unqualified min is available then the run-time behavior might be unaffected.
    # This test acts as a smoke test to ensure the module build is not relying on non‐portable code.
    predictions = torch.randn(128, device="cuda", dtype=torch.float32)
    targets = (torch.randint(0, 2, (128,), device="cuda", dtype=torch.float32) * 2 - 1)
    # Simply call the function – if there were issues with min() the module would not have built.
    out = kernel_module.forward(predictions, targets)
    torch.cuda.synchronize()
    # Validate that output is a single value (the mean hinge loss)
    assert out.numel() == 1, "Expected the forward function to return a scalar tensor."

def test_shape_mismatch_issue(kernel_module):
    # Issue 3: Mismatched number of elements between predictions and targets.
    # The kernel uses predictions.numel() to drive its loop so if targets is smaller, it will read out-of-bound.
    predictions = torch.randn(128, device="cuda", dtype=torch.float32)
    # Create targets with one fewer element
    targets = (torch.randint(0, 2, (127,), device="cuda", dtype=torch.float32) * 2 - 1)
    with pytest.raises(RuntimeError):
        out = kernel_module.forward(predictions, targets)
        torch.cuda.synchronize()

def test_missing_launch_error_checking(kernel_module):
    # Issue 4: The kernel launch is not followed by error checking.
    # One way to trigger an error is to pass a non-contiguous tensor so that CHECK_CONTIGUOUS fails.
    # Although this check is on the CPU side, it simulates a situation where the kernel launch may never happen.
    predictions = torch.randn(128, 1, device="cuda", dtype=torch.float32).t()  # Transpose to make it non-contiguous
    targets = (torch.randint(0, 2, (128,), device="cuda", dtype=torch.float32) * 2 - 1)
    with pytest.raises(RuntimeError) as excinfo:
        out = kernel_module.forward(predictions, targets)
    # Check that the error message indicates the tensor must be contiguous.
    assert "must be contiguous" in str(excinfo.value)
