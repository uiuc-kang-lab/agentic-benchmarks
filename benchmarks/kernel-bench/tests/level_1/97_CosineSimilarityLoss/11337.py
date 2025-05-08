
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the CUDA extension from kernel.cu.
    # Make sure kernel.cu is in the same directory.
    cuda_module = load(
        name="cosine_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test case 1:
# Trigger the issue due to wrong dtype (e.g., float64) because the kernel only supports float32.
def test_input_dtype_error():
    cuda_module = build_kernel()
    batch_size = 128
    D = 4096
    # Create tensors of type float64 instead of float32.
    predictions = torch.randn(batch_size, D, dtype=torch.float64, device='cuda')
    targets = torch.randn(batch_size, D, dtype=torch.float64, device='cuda')
    with pytest.raises(RuntimeError):
        # Expect the TORCH_CHECK in the kernel launcher to trigger an error.
        _ = cuda_module.forward(predictions, targets)

# Test case 2:
# Trigger the issue due to non-contiguous input.
# The kernel assumes contiguous memory but many PyTorch operations result in non-contiguous tensors.
def test_non_contiguous_input():
    cuda_module = build_kernel()
    batch_size = 128
    D = 4096

    # Create contiguous tensors.
    predictions = torch.randn(batch_size, D, device='cuda', dtype=torch.float32)
    targets = torch.randn(batch_size, D, device='cuda', dtype=torch.float32)

    # Make predictions non-contiguous by transposing a larger tensor and then slicing.
    big_tensor = torch.randn(batch_size, D*2, device='cuda', dtype=torch.float32)
    # Create a non-contiguous view by selecting every other column.
    predictions_noncont = big_tensor[:, ::2]
    # Force non-contiguity check
    assert not predictions_noncont.is_contiguous(), "Test setup error: predictions_noncont should be non-contiguous."

    # Use the same tensor for targets to trigger the error.
    targets_noncont = big_tensor[:, ::2]
    assert not targets_noncont.is_contiguous(), "Test setup error: targets_noncont should be non-contiguous."

    # Compute loss using the CUDA kernel.
    loss_kernel = cuda_module.forward(predictions_noncont, targets_noncont).item()

    # Compute loss using PyTorch's cosine similarity loss.
    cosine_sim = torch.nn.functional.cosine_similarity(predictions_noncont, targets_noncont, dim=1)
    loss_ref = torch.mean(1 - cosine_sim).item()

    # Since the kernel does not account for non-contiguous memory layouts,
    # the computed loss is likely to be different than the reference.
    # We require that they are not almost equal.
    assert abs(loss_kernel - loss_ref) > 1e-3, (
        "Kernel did not trigger an error with non-contiguous inputs even though its assumptions are violated."
    )
