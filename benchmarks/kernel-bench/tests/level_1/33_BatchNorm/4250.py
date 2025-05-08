
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the CUDA extension (assumes kernel.cu is in the current directory)
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Kernel only supports float32.
def test_non_float_input():
    cuda_module = build_kernel()
    # Create double precision tensors. The kernel uses data_ptr<float>(), so this should cause memory misinterpretation or error.
    N, C, H, W = 2, 2, 4, 4
    x = torch.randn(N, C, H, W, dtype=torch.double, device='cuda')
    weight = torch.ones(C, dtype=torch.double, device='cuda')
    bias = torch.zeros(C, dtype=torch.double, device='cuda')
    running_mean = torch.zeros(C, dtype=torch.double, device='cuda')
    running_var = torch.ones(C, dtype=torch.double, device='cuda')
    # Expect the kernel to either crash or throw a runtime error
    with pytest.raises(RuntimeError):
        cuda_module.forward(x, weight, bias, running_mean, running_var, True, 0.1, 1e-5)

# Issue 2: Kernel only supports 4D input tensors.
def test_non_4d_input():
    cuda_module = build_kernel()
    # Create a 3D input tensor instead of 4D.
    x = torch.randn(2, 4, 4, device='cuda', dtype=torch.float32)
    # For a 3D input, we cannot extract H and W properly.
    # We create 1D weight, bias, running_mean and running_var assuming C should be at index 1.
    weight = torch.ones(4, dtype=torch.float32, device='cuda')
    bias = torch.zeros(4, dtype=torch.float32, device='cuda')
    running_mean = torch.zeros(4, dtype=torch.float32, device='cuda')
    running_var = torch.ones(4, dtype=torch.float32, device='cuda')
    with pytest.raises(IndexError):
        # This should fail because the kernel uses input.size(2) and input.size(3)
        cuda_module.forward(x, weight, bias, running_mean, running_var, True, 0.1, 1e-5)

# Issue 3: Fixed shared-memory size usage which assumes blockDim.x/warpSize <= 32.
@pytest.mark.xfail(reason="Kernel fixed shared memory arrays assumed blockDim.x/32 <= 32, cannot trigger with default host configuration.")
def test_excessive_threads():
    cuda_module = build_kernel()
    # Here, we try to mimic a scenario (by creating a tensor with a large number of elements so that
    # a different launch configuration might be used) even though the host code fixes threads=256.
    # In a general implementation, if one were to launch the kernel with more threads per block,
    # the statically allocated shared arrays (size 32) would be insufficient.
    N, C, H, W = 2, 64, 32, 32
    x = torch.randn(N, C, H, W, dtype=torch.float32, device='cuda')
    weight = torch.ones(C, dtype=torch.float32, device='cuda')
    bias = torch.zeros(C, dtype=torch.float32, device='cuda')
    running_mean = torch.zeros(C, dtype=torch.float32, device='cuda')
    running_var = torch.ones(C, dtype=torch.float32, device='cuda')
    # Although host always launches with 256 threads per block, this test is marked xfail to indicate that
    # if someone changes the configuration, the fixed shared memory size may lead to errors.
    out = cuda_module.forward(x, weight, bias, running_mean, running_var, True, 0.1, 1e-5)
    # No assertion; the expectation is that when misconfigured, the kernel run might crash.

# Issue 4: Lack of post-launch error checking.
def test_kernel_launch_error():
    cuda_module = build_kernel()
    # Create an input tensor with H or W equal to 0 which will cause numElements==0 and a division by zero.
    N, C, H, W = 2, 2, 0, 0
    x = torch.randn(N, C, H, W, dtype=torch.float32, device='cuda')
    weight = torch.ones(C, dtype=torch.float32, device='cuda')
    bias = torch.zeros(C, dtype=torch.float32, device='cuda')
    running_mean = torch.zeros(C, dtype=torch.float32, device='cuda')
    running_var = torch.ones(C, dtype=torch.float32, device='cuda')
    # Since the kernel does not check for numElements==0, a division-by-zero may occur.
    with pytest.raises(Exception):
        cuda_module.forward(x, weight, bias, running_mean, running_var, True, 0.1, 1e-5)

# Issue 5: Kernel assumes input tensors are contiguous.
def test_non_contiguous_input():
    cuda_module = build_kernel()
    N, C, H, W = 2, 2, 4, 4
    x = torch.randn(N, C, H, W, dtype=torch.float32, device='cuda')
    # Make the input non-contiguous (e.g. via a transpose)
    x_noncontig = x.transpose(1, 2)
    # Adjust weight and others accordingly (still of length equal to C, though the channel dimension might not be at dim1 now)
    weight = torch.ones(x_noncontig.size(1), dtype=torch.float32, device='cuda')
    bias = torch.zeros(x_noncontig.size(1), dtype=torch.float32, device='cuda')
    running_mean = torch.zeros(x_noncontig.size(1), dtype=torch.float32, device='cuda')
    running_var = torch.ones(x_noncontig.size(1), dtype=torch.float32, device='cuda')
    with pytest.raises(RuntimeError):
        cuda_module.forward(x_noncontig, weight, bias, running_mean, running_var, True, 0.1, 1e-5)
