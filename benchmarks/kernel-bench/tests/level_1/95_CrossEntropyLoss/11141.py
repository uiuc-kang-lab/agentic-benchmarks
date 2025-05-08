
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    return load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

# Issue 1: Non‑contiguous predictions tensor causing incorrect behavior.
def test_non_contiguous_predictions():
    batch_size = 4096
    num_classes = 10
    # Create a contiguous tensor then make it non-contiguous via transpose.
    original = torch.randn(batch_size, num_classes, device="cuda", dtype=torch.float32)
    predictions = original.t()  # now shape is (num_classes, batch_size), non‑contiguous.
    # Correct the shape by transposing back when computing reference loss.
    targets = torch.randint(0, num_classes, (batch_size,), device="cuda", dtype=torch.int64)
    kernel_module = build_kernel()
    # Although predictions.t() is the same shape as 'original', the underlying memory layout is different,
    # and the kernel (which directly uses data_ptr) will misinterpret the data.
    loss_kernel = kernel_module.forward(predictions.t(), targets)
    loss_ref = torch.nn.functional.cross_entropy(original, targets)
    # Force a synchronization to ensure the kernel finished.
    torch.cuda.synchronize()
    # Expect a significant difference due to the wrong memory layout.
    with pytest.raises(AssertionError):
        assert torch.allclose(loss_kernel, loss_ref, atol=1e-4), "Loss from non-contiguous input should differ from reference."

# Issue 2: Incorrect dtype for targets (using int32 instead of the required int64).
def test_wrong_target_dtype():
    batch_size = 4096
    num_classes = 10
    predictions = torch.randn(batch_size, num_classes, device="cuda", dtype=torch.float32)
    # Intentionally use int32 rather than int64 for targets.
    targets = torch.randint(0, num_classes, (batch_size,), device="cuda", dtype=torch.int32)
    kernel_module = build_kernel()
    with pytest.raises(RuntimeError):
        kernel_module.forward(predictions, targets)

# Issue 3: Out‑of‑bound target indices.
def test_invalid_target_index():
    batch_size = 4096
    num_classes = 10
    predictions = torch.randn(batch_size, num_classes, device="cuda", dtype=torch.float32)
    targets = torch.randint(0, num_classes, (batch_size,), device="cuda", dtype=torch.int64)
    # Set one of the target indices to an invalid value (num_classes is not a valid index)
    targets[0] = num_classes
    kernel_module = build_kernel()
    with pytest.raises(Exception):
        loss = kernel_module.forward(predictions, targets)
        torch.cuda.synchronize()  # force error propagation

# Issue 4: Invalid tensor dimensions for predictions (the kernel expects a 2D tensor).
def test_invalid_prediction_dimension():
    batch_size = 4096
    num_classes = 10
    # Create a 3D predictions tensor, which the kernel should reject.
    predictions = torch.randn(batch_size, num_classes, 1, device="cuda", dtype=torch.float32)
    targets = torch.randint(0, num_classes, (batch_size,), device="cuda", dtype=torch.int64)
    kernel_module = build_kernel()
    with pytest.raises(RuntimeError):
        kernel_module.forward(predictions, targets)
