
import torch
import pytest
from torch.utils.cpp_extension import load
import os

# Build the CUDA extension from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Expected hinge loss: mean(max(0,1-pred*target))
def hinge_loss_ref(predictions, targets):
    return torch.mean(torch.clamp(1 - predictions * targets, min=0))

@pytest.fixture(scope="module")
def cuda_kernel():
    mod = build_kernel()
    return mod

# Issue 1: The kernel does not check the dtype.
# When using a tensor of type double, the kernel will misinterpret its data.
def test_dtype_mismatch(cuda_kernel):
    # Create double tensors (64-bit) on CUDA.
    batch_size = 128
    # Even though our python-level loss uses clamp, the kernel will read the bytes wrongly.
    predictions = torch.randn(batch_size, dtype=torch.double, device='cuda').contiguous()
    # Ensure targets are double as well.
    targets = (torch.randint(0, 2, (batch_size,), device='cuda').double() * 2 - 1).contiguous()
    # Compute kernel output.
    result = cuda_kernel.forward(predictions, targets)
    # Compute reference using correct dtype by converting to float
    predictions_float = predictions.float()
    targets_float = targets.float()
    ref_result = hinge_loss_ref(predictions_float, targets_float)
    # They should not be equal because the kernel misinterprets the data.
    assert not torch.allclose(result, ref_result, atol=1e-5), \
        f"Kernel unexpectedly produced correct results for double input!"

# Issue 2: The kernel does not verify that predictions and targets have the same number of elements.
def test_shape_mismatch(cuda_kernel):
    batch_size = 128
    predictions = torch.randn(batch_size, device='cuda', dtype=torch.float32).contiguous()
    # Create targets with fewer elements.
    targets = (torch.randint(0, 2, (batch_size // 2,), device='cuda', dtype=torch.float32) * 2 - 1).contiguous()
    # Expect an error when launching the kernel since predictions.numel() != targets.numel().
    with pytest.raises(RuntimeError):
        result = cuda_kernel.forward(predictions, targets)
        # Force synchronization to catch asynchronous errors.
        torch.cuda.synchronize()

# Issue 3: For very large inputs, the kernel uses a truncation on the grid dimension which can leave some elements unprocessed.
def test_large_input_incomplete_processing(cuda_kernel):
    # We pick n such that computed blocks would exceed the maximum grid dimension.
    # Under the current kernel, when n > (max_blocks * block_size) some elements are left unprocessed.
    # The maximum number of blocks is 65535. For large n the adaptive policy selects block_size = 512.
    max_blocks = 65535
    block_size = 512
    n = max_blocks * block_size + 1  # One more than the maximum handled elements

    # Create input tensors filled with 1.0.
    # For targets, we choose -1.0 so that hinge loss becomes fmax(0, 1 - (1 * -1)) = 2.
    predictions = torch.ones(n, device='cuda', dtype=torch.float32).contiguous()
    targets = -torch.ones(n, device='cuda', dtype=torch.float32).contiguous()

    # The correct hinge loss mean computed on CPU would be 2.
    ref = hinge_loss_ref(predictions, targets)
    result = cuda_kernel.forward(predictions, targets)
    # If the kernel fails to process the last element(s), the computed mean will not be equal to the reference.
    # We expect a significant difference.
    assert not torch.allclose(result, ref, atol=1e-5), \
        f"Kernel computed mean {result} close to expected {ref} despite large input causing grid truncation."

# Issue 4: No error checking after the kernel launch.
# A typical way to trigger an error is to pass non-CUDA tensors.
def test_non_cuda_input(cuda_kernel):
    batch_size = 128
    predictions = torch.randn(batch_size, dtype=torch.float32).contiguous()  # CPU tensor
    targets = (torch.randint(0, 2, (batch_size,), dtype=torch.float32) * 2 - 1).contiguous()  # CPU tensor
    with pytest.raises(RuntimeError, match="must be a CUDA tensor"):
        cuda_kernel.forward(predictions, targets)
