
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

def compute_loss_pytorch(predictions, targets):
    # Compute cosine similarity loss as in the original PyTorch Model.
    # Loss = mean(1 - cosine_similarity)
    cos_sim = torch.nn.functional.cosine_similarity(predictions, targets, dim=1)
    return torch.mean(1.0 - cos_sim)

# Issue 1: Non-contiguous tensor input
def test_non_contiguous_tensor():
    # Create contiguous inputs first.
    N, D = 128, 4096
    base_pred = torch.randn(N, D * 2, device='cuda', dtype=torch.float32)
    base_target = torch.randn(N, D * 2, device='cuda', dtype=torch.float32)
    # Create non-contiguous slices by taking every other element.
    predictions = base_pred[:, ::2]
    targets = base_target[:, ::2]
    assert not predictions.is_contiguous(), "Test tensor 'predictions' is unexpectedly contiguous"
    assert not targets.is_contiguous(), "Test tensor 'targets' is unexpectedly contiguous"
    
    my_module = build_kernel()
    output_kernel = my_module.forward(predictions, targets)
    # Compute expected output from PyTorch reference.
    output_ref = compute_loss_pytorch(predictions, targets)
    # This test is meant to reveal issues if the kernel assumes contiguous memory.
    # We check if the difference is larger than a small tolerance.
    torch.cuda.synchronize()
    assert not torch.allclose(output_kernel, output_ref, atol=1e-4), \
        "Kernel should have issues with non-contiguous inputs but produced a close match."

# Issue 2: Fixed warp mask in shuffle reduction
def test_fixed_warp_mask():
    # Using a small value of D to create a scenario where many threads get no work.
    # With a large block size, many threads will be idle and the fixed mask might lead to reduction errors.
    N, D = 128, 10  # D is small relative to block size (default block_size=256)
    predictions = torch.randn(N, D, device='cuda', dtype=torch.float32)
    targets = torch.randn(N, D, device='cuda', dtype=torch.float32)
    
    my_module = build_kernel()
    output_kernel = my_module.forward(predictions, targets)
    output_ref = compute_loss_pytorch(predictions, targets)
    torch.cuda.synchronize()
    # We expect that the fixed mask might cause an error in reduction on these inputs.
    assert not torch.allclose(output_kernel, output_ref, atol=1e-4), \
        "Kernel’s warp reduction (with fixed mask) should yield error with small D relative to block size."

# Issue 3: Irregular D not matching block dimensions assumptions.
def test_irregular_dimension():
    # Choose D that is not a multiple of the block size (256) to stress the assumptions.
    N, D = 128, 1000  # 1000 is not a multiple of 256.
    predictions = torch.randn(N, D, device='cuda', dtype=torch.float32)
    targets = torch.randn(N, D, device='cuda', dtype=torch.float32)
    
    my_module = build_kernel()
    output_kernel = my_module.forward(predictions, targets)
    output_ref = compute_loss_pytorch(predictions, targets)
    torch.cuda.synchronize()
    # This test is aimed at detecting potential load imbalance or reduction logic errors.
    # We check for a noticeable discrepancy.
    assert not torch.allclose(output_kernel, output_ref, atol=1e-4), \
        "Kernel output with irregular D should not match the reference due to reduction issues."

# Issue 4: Lack of kernel launch error checking / crash for invalid tensor type.
def test_invalid_dtype():
    # Pass tensor with dtype float64 rather than float32.
    N, D = 128, 4096
    predictions = torch.randn(N, D, device='cuda', dtype=torch.float64)
    targets = torch.randn(N, D, device='cuda', dtype=torch.float64)
    
    my_module = build_kernel()
    with pytest.raises(RuntimeError):
        # The extension function checks for dtype float32 and should raise an error.
        my_module.forward(predictions, targets)
