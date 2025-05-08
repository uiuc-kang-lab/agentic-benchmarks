
import pytest
import torch
from torch.utils.cpp_extension import load

# Utility function to build the CUDA extension module
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Test case 1: Trigger the wrong shared-memory allocation/indexing by using num_classes > warp size.
def test_large_num_classes():
    # Use a larger number of classes than warp size (e.g., 64) to trigger potential out-of-bound shared-memory accesses.
    batch_size = 128
    num_classes = 64  # >32, the warp size
    predictions = torch.randn(batch_size, num_classes, device="cuda", dtype=torch.float32)
    # Use valid target indices.
    targets = torch.randint(0, num_classes, (batch_size,), device="cuda", dtype=torch.int64)
    
    my_module = build_kernel()
    # Compute loss using the custom CUDA kernel.
    loss_kernel = my_module.forward(predictions, targets)
    torch.cuda.synchronize()
    # Compute loss using PyTorch built-in function.
    loss_ref = torch.nn.functional.cross_entropy(predictions, targets).mean()
    # Since the kernel has indexing errors with large num_classes,
    # we expect the value to be different from the correct value.
    assert not torch.allclose(loss_kernel, loss_ref, atol=1e-4), \
        f"Kernel loss unexpectedly matches reference loss. Kernel loss: {loss_kernel.item()}, Reference loss: {loss_ref.item()}"

# Test case 2: Trigger the divergent __syncthreads issue by using a batch size smaller than warps per block.
def test_small_batch_divergence():
    # Use a batch_size that is smaller than the number of warps launched in one block.
    # For example, with threads_per_block=128 (i.e., 4 warps) set in the kernel launch,
    # if batch_size = 2 then two warps will return early and others will still call __syncthreads().
    batch_size = 2
    num_classes = 10
    predictions = torch.randn(batch_size, num_classes, device="cuda", dtype=torch.float32)
    targets = torch.randint(0, num_classes, (batch_size,), device="cuda", dtype=torch.int64)
    
    my_module = build_kernel()
    with pytest.raises(RuntimeError):
        # This call is expected to hang or error out due to divergent __syncthreads().
        loss_kernel = my_module.forward(predictions, targets)
        torch.cuda.synchronize()

# Test case 3: Pass predictions tensor with wrong dtype (not Float32) to trigger a TORCH_CHECK error.
def test_invalid_prediction_dtype():
    batch_size = 128
    num_classes = 10
    predictions = torch.randn(batch_size, num_classes, device="cuda", dtype=torch.float64)  # wrong dtype
    targets = torch.randint(0, num_classes, (batch_size,), device="cuda", dtype=torch.int64)
    
    my_module = build_kernel()
    with pytest.raises(RuntimeError) as excinfo:
        my_module.forward(predictions, targets)
    assert "predictions must be Float32 tensor" in str(excinfo.value)

# Test case 4: Pass targets tensor with wrong dtype (not Int64) to trigger a TORCH_CHECK error.
def test_invalid_target_dtype():
    batch_size = 128
    num_classes = 10
    predictions = torch.randn(batch_size, num_classes, device="cuda", dtype=torch.float32)
    targets = torch.randint(0, num_classes, (batch_size,), device="cuda", dtype=torch.int32)  # wrong dtype
    
    my_module = build_kernel()
    with pytest.raises(RuntimeError) as excinfo:
        my_module.forward(predictions, targets)
    assert "targets must be Int64 tensor" in str(excinfo.value)
