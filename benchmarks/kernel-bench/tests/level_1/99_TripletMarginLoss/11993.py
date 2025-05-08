
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="triplet_loss_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# A helper function that computes the reference triplet margin loss using PyTorch's built-in module.
def reference_loss(anchor, positive, negative, margin):
    loss_fn = torch.nn.TripletMarginLoss(margin=margin, reduction="mean")
    return loss_fn(anchor, positive, negative)

# Issue 1: Non-contiguous input tensors are assumed to be contiguous.
def test_non_contiguous_input():
    # Create contiguous inputs first.
    batch_size = 128
    feat_size = 4096
    margin = 1.0
    anchor = torch.randn(batch_size*2, feat_size, device="cuda")[::2]
    positive = torch.randn(batch_size*2, feat_size, device="cuda")[::2]
    negative = torch.randn(batch_size*2, feat_size, device="cuda")[::2]
    
    # Verify that the inputs are non-contiguous.
    assert not anchor.is_contiguous(), "Input anchor tensor is unexpectedly contiguous."

    # Build the kernel module.
    mod = build_kernel()
    
    # Run the CUDA kernel.
    try:
        loss_kernel = mod.forward(anchor, positive, negative, margin)
    except Exception as e:
        pytest.fail(f"Kernel failed with non-contiguous input: {e}")
    
    # Compute reference loss (forcing contiguous input).
    loss_ref = reference_loss(anchor.contiguous(), positive.contiguous(), negative.contiguous(), margin)
    
    # Because the kernel internally assumes contiguous layout, its result is likely incorrect.
    # We test that the kernel result deviates significantly from the reference.
    assert not torch.allclose(loss_kernel, loss_ref, atol=1e-3), \
        f"Kernel did not trigger an error with non-contiguous inputs; expected a deviation from reference loss."

# Issue 2: Mismatched shapes among the three tensors are not checked.
def test_mismatched_shapes():
    batch_size = 128
    feat_size = 4096
    margin = 1.0
    
    anchor = torch.randn(batch_size, feat_size, device="cuda")
    positive = torch.randn(batch_size, feat_size, device="cuda")
    # Introduce an error: mismatched negative tensor with an extra feature.
    negative = torch.randn(batch_size, feat_size + 1, device="cuda")
    
    mod = build_kernel()
    
    with pytest.raises(RuntimeError):
        # The kernel is expected to run into an out-of-bound error (or similar error)
        # due to mismatched shapes.
        _ = mod.forward(anchor, positive, negative, margin)
        torch.cuda.synchronize()

# Issue 3: The kernel is written for float32 tensors only.
def test_incorrect_dtype():
    batch_size = 128
    feat_size = 4096
    margin = 1.0
    
    # Create inputs with dtype float64 (double)
    anchor = torch.randn(batch_size, feat_size, device="cuda", dtype=torch.double)
    positive = torch.randn(batch_size, feat_size, device="cuda", dtype=torch.double)
    negative = torch.randn(batch_size, feat_size, device="cuda", dtype=torch.double)
    
    mod = build_kernel()
    
    # Running the kernel on double tensors will reinterpret memory.
    try:
        loss_kernel = mod.forward(anchor, positive, negative, margin)
    except Exception as e:
        pytest.fail(f"Kernel raised an unexpected exception when using double tensors: {e}")
    
    # Compute reference loss using float32 tensors (after explicit conversion).
    anchor_f = anchor.float()
    positive_f = positive.float()
    negative_f = negative.float()
    loss_ref = reference_loss(anchor_f, positive_f, negative_f, margin)
    
    # The result from the kernel run is likely numerically very different.
    assert not torch.allclose(loss_kernel, loss_ref, atol=1e-3), \
        "Kernel output for double input is unexpectedly close to reference loss, though it should be incorrect."

