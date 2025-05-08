
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

# 1. Test to trigger wrong data type (using torch.double instead of torch.float32)
def test_tensor_dtype_error():
    my_module = build_kernel()
    # Create double tensors (instead of float32) - kernel assumes float32.
    predictions = torch.randn(128, 1, device="cuda", dtype=torch.double)
    targets = (torch.randint(0, 2, (128, 1), device="cuda", dtype=torch.double) * 2 - 1)
    with pytest.raises(RuntimeError):
        # This call should error because the kernel uses data_ptr<float>().
        my_module.forward(predictions, targets)

# 2. Test for mismatched shapes: predictions and targets with different number of elements.
def test_mismatched_shapes():
    my_module = build_kernel()
    # Create a predictions tensor with 256 elements and a targets tensor with 128 elements.
    predictions = torch.randn(256, device="cuda", dtype=torch.float32)
    targets = (torch.randint(0, 2, (128,), device="cuda", dtype=torch.float32) * 2 - 1)
    # Although the kernel only uses predictions.numel() for loop, accessing targets beyond its limit is undefined.
    # We expect a CUDA error (e.g., out-of-bounds memory access).
    with pytest.raises(RuntimeError):
        out = my_module.forward(predictions, targets)
        torch.cuda.synchronize()

# 3. Test for non-contiguous tensor input.
def test_non_contiguous_input():
    my_module = build_kernel()
    predictions = torch.randn(128, 1, device="cuda", dtype=torch.float32)
    targets = (torch.randint(0, 2, (128, 1), device="cuda", dtype=torch.float32) * 2 - 1)
    # Make predictions non-contiguous by transposing.
    predictions_non_contig = predictions.t()
    with pytest.raises(RuntimeError):
        my_module.forward(predictions_non_contig, targets)

# 4. Test for kernel launch errors. Without proper error checking inside the CUDA code,
# we simulate the possibility by launching a kernel with an extremely large number of elements.
def test_kernel_launch_error():
    my_module = build_kernel()
    # Warning: This test is platform and GPU dependent and might take a long time or behave differently.
    # We attempt to create a scenario that may trigger a kernel launch error.
    size = 2**29  # A very large tensor (adjust as needed to stress the GPU)
    try:
        predictions = torch.randn(size, device="cuda", dtype=torch.float32)
        targets = (torch.randint(0, 2, (size,), device="cuda", dtype=torch.float32) * 2 - 1)
    except RuntimeError:
        pytest.skip("Not enough memory to run the kernel launch error test.")
        
    with pytest.raises(RuntimeError):
        out = my_module.forward(predictions, targets)
        torch.cuda.synchronize()
