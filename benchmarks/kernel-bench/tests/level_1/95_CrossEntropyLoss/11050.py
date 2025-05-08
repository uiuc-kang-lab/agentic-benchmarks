
import pytest
import torch
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

# Test case 1: Wrong predictions tensor data type (non-float32)
def test_wrong_dtype_predictions():
    module = build_kernel()
    batch_size, num_classes = 16, 4
    # Create predictions with double precision instead of float32
    predictions = torch.randn(batch_size, num_classes, dtype=torch.float64, device="cuda")
    targets = torch.randint(0, num_classes, (batch_size,), device="cuda")
    with pytest.raises(RuntimeError, match="predictions must be a Float32 tensor"):
        module.forward(predictions, targets)

# Test case 2: Wrong targets tensor data type (non-int64)
def test_wrong_dtype_targets():
    module = build_kernel()
    batch_size, num_classes = 16, 4
    predictions = torch.randn(batch_size, num_classes, dtype=torch.float32, device="cuda")
    # Create targets with wrong type (e.g. float32)
    targets = torch.randint(0, num_classes, (batch_size,), device="cuda").to(torch.float32)
    with pytest.raises(RuntimeError, match="targets must be Int64 tensor"):
        module.forward(predictions, targets)

# Test case 3: Wrong dimension for predictions (should be 2D)
def test_wrong_dimension_predictions():
    module = build_kernel()
    batch_size, num_classes = 16, 4
    # Create a 3D tensor instead of 2D (e.g. add an extra dimension)
    predictions = torch.randn(batch_size, num_classes, 1, dtype=torch.float32, device="cuda")
    targets = torch.randint(0, num_classes, (batch_size,), device="cuda")
    with pytest.raises(RuntimeError, match="predictions must be a 2D tensor"):
        module.forward(predictions, targets)

# Test case 4: Wrong dimension for targets (should be 1D)
def test_wrong_dimension_targets():
    module = build_kernel()
    batch_size, num_classes = 16, 4
    predictions = torch.randn(batch_size, num_classes, dtype=torch.float32, device="cuda")
    # Create a 2D targets tensor
    targets = torch.randint(0, num_classes, (batch_size, 1), device="cuda")
    with pytest.raises(RuntimeError, match="targets must be a 1D tensor"):
        module.forward(predictions, targets)

# Test case 5: Mismatched batch sizes between predictions and targets
def test_mismatched_batch_size():
    module = build_kernel()
    batch_size, num_classes = 16, 4
    predictions = torch.randn(batch_size, num_classes, dtype=torch.float32, device="cuda")
    # Create targets with a different batch size
    targets = torch.randint(0, num_classes, (batch_size + 1,), device="cuda")
    with pytest.raises(RuntimeError, match="targets must have same batch size as predictions"):
        module.forward(predictions, targets)

# Test case 6: Out-of-bound target value
def test_out_of_bound_target():
    module = build_kernel()
    batch_size, num_classes = 16, 4
    predictions = torch.randn(batch_size, num_classes, dtype=torch.float32, device="cuda")
    # Set one target to an invalid value (num_classes is out-of-bound)
    targets = torch.randint(0, num_classes, (batch_size,), device="cuda")
    targets[0] = num_classes  # invalid target index
    # This may cause a CUDA device error (or undefined behavior) which should be caught.
    with pytest.raises(RuntimeError):
        module.forward(predictions, targets)

# Test case 7: Non-contiguous predictions tensor
def test_non_contiguous_predictions():
    module = build_kernel()
    batch_size, num_classes = 1024, 10
    # Create a contiguous predictions tensor
    base = torch.randn(batch_size * 2, num_classes, dtype=torch.float32, device="cuda")
    # Use advanced indexing to create a non-contiguous tensor with the correct shape
    predictions = base[::2]  # This slicing makes it non-contiguous
    # Verify that the predictions tensor is indeed non-contiguous
    assert not predictions.is_contiguous()
    targets = torch.randint(0, num_classes, (batch_size,), device="cuda")

    # Compute loss using our kernel
    loss_kernel = module.forward(predictions, targets)

    # Compute reference loss by forcing predictions to be contiguous
    loss_ref = torch.nn.functional.cross_entropy(predictions.contiguous(), targets)
    # Due to the non-contiguous layout, the kernel will access wrong memory locations.
    # Hence, the loss computed by the kernel is expected to differ significantly from the reference.
    assert not torch.allclose(loss_kernel, loss_ref, atol=1e-4), (
        f"Kernel loss ({loss_kernel.item()}) unexpectedly matches reference loss ({loss_ref.item()}) "
        "despite non-contiguous input."
    )
