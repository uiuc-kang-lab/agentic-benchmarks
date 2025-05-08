
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="hinge_loss_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test case 1: Wrong dtype (not float32).
def test_wrong_dtype():
    my_module = build_kernel()
    # Create tensors using double precision.
    predictions = torch.randn(128, 1, dtype=torch.double, device='cuda')
    targets = (torch.randint(0, 2, (128,), device='cuda', dtype=torch.int64).float() * 2 - 1).double()
    with pytest.raises(RuntimeError, match="must be a CUDA tensor"):
        # The CHECK_INPUT macros assume float and contiguous so using a different dtype may trigger errors later.
        my_module.forward(predictions, targets)

# Test case 2: Misaligned input memory
def test_misaligned_memory():
    my_module = build_kernel()
    # Create a tensor with one extra element and then slice to force a misaligned pointer.
    # Even though the slice is still 'contiguous' in PyTorch, the storage offset makes the pointer misaligned.
    base_pred = torch.randn(129, dtype=torch.float32, device='cuda')
    base_targ = (torch.randint(0, 2, (129,), device='cuda', dtype=torch.int64).float() * 2 - 1)
    # Slicing off one element may cause the underlying pointer not to be 16-byte aligned.
    predictions = base_pred[1:].clone()  # clone to get a new allocation with the same misaligned offset offset
    targets = base_targ[1:].clone()
    # Compute kernel result and compare to the PyTorch implementation.
    # In case of misalignment, the kernel may produce an incorrect result.
    result_kernel = my_module.forward(predictions, targets)
    result_ref = torch.mean(torch.clamp(1 - predictions * targets, min=0))
    # We expect the two results to match; if not, this indicates the kernel mishandles misaligned pointers.
    assert torch.allclose(result_kernel, result_ref, atol=1e-5), \
         f"Kernel result {result_kernel} does not match reference result {result_ref} for misaligned input."

# Test case 3: Non-contiguous tensor
def test_non_contiguous_tensor():
    my_module = build_kernel()
    # Create a contiguous tensor then take a transpose to break contiguity.
    predictions = torch.randn(8, 16, dtype=torch.float32, device='cuda')
    targets = ((torch.randint(0, 2, (8, 16), device='cuda', dtype=torch.int64).float() * 2) - 1)
    non_contig_pred = predictions.t()  # transpose makes it non-contiguous
    non_contig_targ = targets.t()
    with pytest.raises(RuntimeError, match="must be contiguous"):
        my_module.forward(non_contig_pred, non_contig_targ)

# Test case 4: Tensor with complex shape (non-flat layout)
def test_complex_tensor():
    my_module = build_kernel()
    # Create a tensor that is contiguous but multi-dimensional
    predictions = torch.randn(4, 32, dtype=torch.float32, device='cuda')
    targets = ((torch.randint(0, 2, (4, 32), device='cuda', dtype=torch.int64).float() * 2) - 1)
    # Although the tensor is multi-dimensional, it is contiguous and flat access via numel() should work.
    # This test is to ensure that such a general case is handled.
    result_kernel = my_module.forward(predictions, targets)
    result_ref = torch.mean(torch.clamp(1 - predictions * targets, min=0))
    assert torch.allclose(result_kernel, result_ref, atol=1e-5), \
         f"Kernel result {result_kernel} does not match reference result {result_ref} with a complex shape."
