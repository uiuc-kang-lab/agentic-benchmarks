
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Helper function to compute expected KL divergence based on torch.nn.functional.kl_div.
# Note: torch.nn.functional.kl_div with reduction='batchmean' divides the sum by batch_size.
def compute_expected_kl_div(predictions, targets):
    # Compute log(predictions) safely (avoid log(0)) and then use kl_div as defined:
    # F.kl_div expects input: log_predictions, and target: probabilities.
    log_predictions = torch.log(predictions)
    # The torch kl_div computes sum(target * (log(target) - log_predictions))/batch_size.
    batch_size = predictions.shape[0]
    return torch.nn.functional.kl_div(log_predictions, targets, reduction='batchmean')

# Test case 1: Stride Issue
# With a large input tensor the kernel loop should iterate multiple times per thread.
# Because of the incorrect stride, the kernel will omit many elements.
def test_stride_issue():
    # Create a tensor larger than one block can cover in one pass.
    batch_size = 128
    dims = (4096,)  # Same as the original, total elements=128*4096, many blocks needed.
    predictions = torch.randn(batch_size, *dims, device="cuda").softmax(dim=-1)
    targets = torch.randn(batch_size, *dims, device="cuda").softmax(dim=-1)

    module = build_kernel()
    # Call the kernel function. The kernel expects log(predictions)
    output = module.forward(torch.log(predictions), targets)
    torch.cuda.synchronize()

    # Compute reference using torch.nn.functional.kl_div.
    expected = compute_expected_kl_div(predictions, targets)
    # Because the kernel uses an incorrect loop stride, its computed sum will be significantly lower.
    assert not torch.allclose(output.cpu(), expected.cpu(), atol=1e-3), (
        "Test failed to detect stride issue: Kernel output unexpectedly matches reference output."
    )

# Test case 2: Scaling Issue
# Because the kernel divides the sum by the total number of elements rather than the batch size,
# the returned value will be off by a factor (typically the feature dimension).
def test_scaling_issue():
    batch_size = 32
    dims = (128,)  # Total elements = 32*128, whereas expected reduction divides by 32.
    predictions = torch.randn(batch_size, *dims, device="cuda").softmax(dim=-1)
    targets = torch.randn(batch_size, *dims, device="cuda").softmax(dim=-1)

    module = build_kernel()
    output = module.forward(torch.log(predictions), targets)
    torch.cuda.synchronize()

    expected = compute_expected_kl_div(predictions, targets)
    # The scaling error will typically make the kernel output smaller by a factor close to feature-dim size.
    # We assert that the outputs do not match the expected value.
    assert not torch.allclose(output.cpu(), expected.cpu(), atol=1e-3), (
        "Test failed to detect scaling issue: Kernel output unexpectedly matches reference output."
    )

# Test case 3: Input Tensor Type Validation Issue
# Passing a tensor of type float64 should cause issues because the kernel uses float (float32) pointers.
def test_input_tensor_type():
    batch_size = 16
    dims = (64,)
    # Create double tensors instead of float32.
    predictions = torch.randn(batch_size, *dims, device="cuda", dtype=torch.float64).softmax(dim=-1)
    targets = torch.randn(batch_size, *dims, device="cuda", dtype=torch.float64).softmax(dim=-1)

    module = build_kernel()
    with pytest.raises(RuntimeError):
        # The kernel uses data_ptr<float>(), so this should trigger an error or produce a wrong launch.
        module.forward(torch.log(predictions), targets)
    # Note: Depending on the CUDA driver and PyTorch build, this may not always trigger a RuntimeError,
    # but the test is designed to catch unexpected behavior when using a wrong data type.

# Test case 4: Non-contiguous Tensor Issue
# Passing a non-contiguous tensor may cause incorrect results.
def test_non_contiguous_tensor():
    batch_size = 16
    dims = (64, 8)
    predictions = torch.randn(batch_size, *dims, device="cuda", dtype=torch.float32).softmax(dim=-1)
    targets = torch.randn(batch_size, *dims, device="cuda", dtype=torch.float32).softmax(dim=-1)
    # Make tensors non-contiguous by transposing dimensions (if possible)
    predictions_nc = predictions.transpose(0, 1)
    targets_nc = targets.transpose(0, 1)
    
    module = build_kernel()
    with pytest.raises(RuntimeError):
        # The kernel expects contiguous memory; using non-contiguous tensors should cause issues.
        module.forward(torch.log(predictions_nc), targets_nc)
    # Note: If no RuntimeError is raised, then the kernel might work incorrectly due to misinterpreting memory layout.

if __name__ == '__main__':
    pytest.main([__file__])
