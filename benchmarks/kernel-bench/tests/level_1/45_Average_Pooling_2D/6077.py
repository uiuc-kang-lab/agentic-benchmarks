
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="cuda_avg_pool2d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return cuda_module

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_grid_dimension_exceeded():
    """
    This test tries to trigger the grid dimension limitation by setting the number of channels
    (or batch size * channels) higher than CUDA's maximum allowed grid dimension in z.
    For many GPUs the maximum grid dimension in z is 65535.
    Here we set channels high to trigger an error.
    """
    cuda_module = build_kernel()
    # Choose dimensions so that N * C > 65,535. For example, N = 1, C = 70000.
    batch_size = 1
    channels = 70000
    height = 10
    width = 10
    kernel_size = 3
    stride = 3
    padding = 0

    x = torch.randn(batch_size, channels, height, width, device="cuda", dtype=torch.float32)
    # The kernel launch configuration uses grid.z = N * C (here 70000) which likely exceeds the limit.
    with pytest.raises(RuntimeError, match="CUDA error"):
        out = cuda_module.forward(x, kernel_size, stride, padding)
        # Synchronize to force any asynchronous launch errors to be raised.
        torch.cuda.synchronize()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_kernel_size_zero():
    """
    This test triggers the division-by-zero issue by providing a kernel_size of 0.
    The kernel computes the average by dividing by (kernel_size * kernel_size), which leads
    to a division by zero when kernel_size==0.
    """
    cuda_module = build_kernel()
    batch_size = 1
    channels = 3
    height = 10
    width = 10
    kernel_size = 0  # This should trigger a division by zero
    stride = 1
    padding = 0

    x = torch.randn(batch_size, channels, height, width, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        out = cuda_module.forward(x, kernel_size, stride, padding)
        torch.cuda.synchronize()
