
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="triplet_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Helper: reference implementation using PyTorch's native TripletMarginLoss
def reference_triplet_loss(anchor, positive, negative, margin):
    loss_fn = torch.nn.TripletMarginLoss(margin=margin)
    return loss_fn(anchor, positive, negative)

# Issue 1: Hard-coded warp size.
# Although most NVIDIA GPUs have a warp size of 32, a kernel using a fixed 32 might fail 
# on devices with a different warp size or if a test forces an edge-case in feature dimensions.
# We test with a feature dimension that is not a multiple of 32.
def test_warp_size_assumption():
    torch.manual_seed(0)
    batch_size = 128
    feat_size = 45  # not a multiple of 32
    margin = 1.0
    anchor = torch.randn(batch_size, feat_size, device="cuda", dtype=torch.float32)
    positive = torch.randn(batch_size, feat_size, device="cuda", dtype=torch.float32)
    negative = torch.randn(batch_size, feat_size, device="cuda", dtype=torch.float32)
    
    module = build_kernel()
    # The kernel returns the mean loss.
    loss_cuda = module.forward(anchor, positive, negative, margin)
    loss_ref = reference_triplet_loss(anchor, positive, negative, margin)
    
    # With a non-multiple feature dimension the per-warp load balancing may be off.
    # We expect the CUDA result to be different from the PyTorch reference value.
    # (If they match exactly, then the hardcoded warp size did not cause a problem in this test.)
    assert not torch.allclose(loss_cuda, loss_ref, atol=1e-5), \
        "Kernel result matches reference; expected discrepancy due to warp size assumption."

# Issue 2: Lack of support for half precision (float16). 
def test_half_precision_not_supported():
    batch_size = 64
    feat_size = 256
    margin = 1.0
    anchor = torch.randn(batch_size, feat_size, device="cuda", dtype=torch.float16)
    positive = torch.randn(batch_size, feat_size, device="cuda", dtype=torch.float16)
    negative = torch.randn(batch_size, feat_size, device="cuda", dtype=torch.float16)
    
    module = build_kernel()
    with pytest.raises(RuntimeError):
        # Expect an error because AT_DISPATCH_FLOATING_TYPES does not include half precision.
        module.forward(anchor, positive, negative, margin)

# Issue 3: Kernel assumes 2D contiguous inputs.
def test_invalid_tensor_dimensions():
    batch_size = 32
    feat_size = 128
    # Create a tensor with an extra dimension.
    anchor = torch.randn(batch_size, 1, feat_size, device="cuda", dtype=torch.float32)
    positive = torch.randn(batch_size, 1, feat_size, device="cuda", dtype=torch.float32)
    negative = torch.randn(batch_size, 1, feat_size, device="cuda", dtype=torch.float32)
    
    module = build_kernel()
    with pytest.raises(RuntimeError):
        # The kernel computes memory index as sample_idx*feat_size + f.
        # With a 3D tensor the layout is not as expected.
        module.forward(anchor, positive, negative, 1.0)

# Additional test: non-contiguous inputs.
def test_non_contiguous_inputs():
    batch_size = 64
    feat_size = 512
    margin = 1.0
    # Create contiguous inputs.
    anchor = torch.randn(batch_size, feat_size, device="cuda", dtype=torch.float32)
    positive = torch.randn(batch_size, feat_size, device="cuda", dtype=torch.float32)
    negative = torch.randn(batch_size, feat_size, device="cuda", dtype=torch.float32)
    
    # Make them non-contiguous by transposing a dummy extra dimension.
    anchor_nc = anchor.unsqueeze(2).transpose(1,2).squeeze(2)
    positive_nc = positive.unsqueeze(2).transpose(1,2).squeeze(2)
    negative_nc = negative.unsqueeze(2).transpose(1,2).squeeze(2)
    
    # They remain 2D but may have a different memory layout.
    module = build_kernel()
    # The kernel does not check contiguity, so it might produce incorrect results.
    loss_cuda_nc = module.forward(anchor_nc, positive_nc, negative_nc, margin)
    loss_ref = reference_triplet_loss(anchor_nc.contiguous(), positive_nc.contiguous(), negative_nc.contiguous(), margin)
    
    # We expect a discrepancy because of the potential misinterpretation of memory layout.
    assert not torch.allclose(loss_cuda_nc, loss_ref, atol=1e-5), \
        "Kernel result on non-contiguous input matches reference result; expected incorrect behavior due to layout assumptions."
