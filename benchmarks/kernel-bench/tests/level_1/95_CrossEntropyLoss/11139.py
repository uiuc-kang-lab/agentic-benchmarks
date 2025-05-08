
import pytest
import torch
from torch.utils.cpp_extension import load
import os

def build_kernel():
    # Ensure we always compile the kernel from kernel.cu in the current directory.
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    cuda_file = os.path.join(cur_dir, "kernel.cu")
    module = load(
        name="custom_ce_loss",
        sources=[cuda_file],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return module

# Issue 1: Wrong data type for predictions (expecting float32, passing float64)
def test_wrong_dtype():
    torch.cuda.empty_cache()
    batch_size = 128
    num_classes = 10
    # Create predictions with double precision instead of float32.
    predictions = torch.randn(batch_size, num_classes, dtype=torch.double, device='cuda')
    targets = torch.randint(0, num_classes, (batch_size,), device='cuda')
    my_module = build_kernel()
    with pytest.raises(RuntimeError):
        # The kernel does not check for tensor dtype, so the wrong interpretation of bytes
        # should eventually lead to a runtime CUDA error.
        loss = my_module.forward(predictions, targets)
        torch.cuda.synchronize()

# Issue 2: Invalid target index (target value not in [0, num_classes-1])
def test_invalid_target():
    torch.cuda.empty_cache()
    batch_size = 128
    num_classes = 10
    predictions = torch.randn(batch_size, num_classes, dtype=torch.float32, device='cuda')
    # Introduce an out-of-range target index.
    targets = torch.randint(0, num_classes, (batch_size,), device='cuda')
    targets[0] = num_classes  # invalid index; should be in range [0, num_classes-1]
    my_module = build_kernel()
    with pytest.raises(RuntimeError):
        # The kernel does not verify target bounds so this is likely to trigger a CUDA error.
        loss = my_module.forward(predictions, targets)
        torch.cuda.synchronize()
