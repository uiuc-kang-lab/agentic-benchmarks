
import pytest
import torch
from torch.utils.cpp_extension import load

# Build the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="cross_entropy_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test case for Issue 1: Invalid class indices in targets
def test_invalid_target_indices():
    # Create valid predictions
    batch_size = 128
    num_classes = 10
    predictions = torch.randn(batch_size, num_classes, device="cuda", dtype=torch.float32)
    # Create invalid targets where one index is out-of-bound (e.g. equals num_classes)
    targets = torch.randint(0, num_classes, (batch_size,), device="cuda", dtype=torch.int64)
    # Introduce an invalid target on purpose
    targets[0] = num_classes  # invalid index: valid indices are [0, num_classes-1]

    kernel = build_kernel()
    with pytest.raises(RuntimeError):
        # Expect a runtime error because the kernel will try to access logits[target] out-of-bound.
        kernel.forward(predictions, targets)
    torch.cuda.synchronize()

# Test case for Issue 2: Non-contiguous input tensor
def test_non_contiguous_input():
    batch_size = 128
    num_classes = 10

    # Create contiguous predictions and then make a non-contiguous version by transposing
    predictions = torch.randn(batch_size, num_classes, device="cuda", dtype=torch.float32)
    # Transpose to make it non-contiguous and then transpose back to get an equivalent tensor but non-contiguous.
    predictions_non_contiguous = predictions.t().t()  
    # Verify that the tensor is indeed non contiguous.
    assert not predictions_non_contiguous.is_contiguous(), "Tensor is unexpectedly contiguous."

    # Use valid targets
    targets = torch.randint(0, num_classes, (batch_size,), device="cuda", dtype=torch.int64)

    kernel = build_kernel()
    # Forward pass using non-contiguous predictions, which will likely result in an incorrect loss value.
    loss = kernel.forward(predictions_non_contiguous, targets)
    torch.cuda.synchronize()

    # Compute the reference loss using PyTorch's native function with contiguous input.
    loss_ref = torch.nn.functional.cross_entropy(predictions, targets)

    # The result from the kernel should match the reference loss if the kernel handled non-contiguous memory properly.
    # We expect a mismatch here.
    with pytest.raises(AssertionError):
        assert torch.allclose(loss, loss_ref, atol=1e-5), (
            f"Kernel loss {loss.item()} matches reference loss {loss_ref.item()} despite non-contiguous inputs."
        )
