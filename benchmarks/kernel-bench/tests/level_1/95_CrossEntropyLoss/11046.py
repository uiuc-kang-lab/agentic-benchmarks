
import pytest
import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

# Build the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Test 1: Non-contiguous predictions tensor
def test_non_contiguous_predictions():
    # Create contiguous predictions and then a transposed (non-contiguous) version.
    # The kernel assumes a contiguous layout so the computed loss will differ.
    batch_size = 64
    num_classes = 10
    # Create a contiguous tensor and then force non-contiguity by transposing a 2D view.
    predictions = torch.randn(batch_size, num_classes, device="cuda", dtype=torch.float32)
    targets = torch.randint(0, num_classes, (batch_size,), device="cuda", dtype=torch.int64)
    
    # Make predictions non-contiguous by taking its transpose and then transposing back.
    # This yields a non-contiguous tensor with the same shape.
    noncontig = predictions.t().clone().t()
    assert not noncontig.is_contiguous(), "Tensor is unexpectedly contiguous."
    
    module = build_kernel()
    
    # Compute loss using the kernel on contiguous input
    loss_contig = module.forward(predictions, targets)
    # Compute loss using the kernel on non-contiguous input.
    loss_noncontig = module.forward(noncontig, targets)
    
    # Compute the reference loss using PyTorch's cross_entropy (which works for non-contiguous tensors).
    ref_loss = F.cross_entropy(noncontig, targets)
    
    # The result from our kernel will be wrong because the kernel assumed a contiguous layout.
    # We check that our kernel's output (loss_noncontig) differs significantly from the reference.
    assert not torch.allclose(loss_noncontig, ref_loss, atol=1e-5), \
        "Kernel did not fail on non-contiguous input even though it assumes contiguous memory."

# Test 2: Wrong dtype for predictions (e.g. Float64 instead of Float32)
def test_wrong_dtype_predictions():
    batch_size = 64
    num_classes = 10
    predictions = torch.randn(batch_size, num_classes, device="cuda", dtype=torch.float64)
    targets = torch.randint(0, num_classes, (batch_size,), device="cuda", dtype=torch.int64)
    module = build_kernel()
    with pytest.raises(RuntimeError, match="predictions must be a Float32 tensor"):
        # This should trigger the TORCH_CHECK error that ensures predictions are float32.
        module.forward(predictions, targets)

# Test 3: Wrong dtype for targets (e.g. Int32 instead of Int64)
def test_wrong_dtype_targets():
    batch_size = 64
    num_classes = 10
    predictions = torch.randn(batch_size, num_classes, device="cuda", dtype=torch.float32)
    # Create targets with wrong dtype.
    targets = torch.randint(0, num_classes, (batch_size,), device="cuda", dtype=torch.int32)
    module = build_kernel()
    with pytest.raises(RuntimeError, match="targets must be an Int64 tensor"):
        module.forward(predictions, targets)

# Test 4: Out-of-bound target indices
def test_out_of_bound_target():
    batch_size = 64
    num_classes = 10
    predictions = torch.randn(batch_size, num_classes, device="cuda", dtype=torch.float32)
    # Introduce an out-of-bound index (e.g., equal to num_classes)
    targets = torch.randint(0, num_classes, (batch_size,), device="cuda", dtype=torch.int64)
    targets[0] = num_classes  # deliberately set an invalid class index
    module = build_kernel()
    # We expect undefined behavior. In our PyTorch host launcher the TORCH_CHECK doesn't catch this,
    # so the kernel is likely to produce a NaN loss for the sample with the invalid target.
    loss = module.forward(predictions, targets)
    # Since one loss is expected to be NaN, the overall mean should be NaN.
    assert torch.isnan(loss), "Kernel did not produce NaN for out-of-bound target index."

# Test 5: Incorrect dimensionality for predictions (not 2D)
def test_wrong_dim_predictions():
    # Provide a 3D tensor for predictions.
    batch_size = 64
    num_classes = 10
    predictions = torch.randn(batch_size, num_classes, 1, device="cuda", dtype=torch.float32)
    targets = torch.randint(0, num_classes, (batch_size,), device="cuda", dtype=torch.int64)
    module = build_kernel()
    with pytest.raises(RuntimeError, match="predictions must be a 2D tensor"):
        module.forward(predictions, targets)

# Test 6: Incorrect dimensionality for targets (not 1D)
def test_wrong_dim_targets():
    batch_size = 64
    num_classes = 10
    predictions = torch.randn(batch_size, num_classes, device="cuda", dtype=torch.float32)
    # Provide a 2D tensor for targets.
    targets = torch.randint(0, num_classes, (batch_size, 1), device="cuda", dtype=torch.int64)
    module = build_kernel()
    with pytest.raises(RuntimeError, match="targets must be a 1D tensor"):
        module.forward(predictions, targets)
