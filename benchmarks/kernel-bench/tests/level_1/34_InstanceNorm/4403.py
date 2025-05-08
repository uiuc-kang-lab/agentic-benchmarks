
import pytest
import torch
from torch.utils.cpp_extension import load
import warnings

def build_kernel():
    # Build the CUDA extension from the provided kernel source file "kernel.cu"
    module = load(
        name="instance_norm_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return module

# Issue 2: Test that a tensor with very large H and W causes the kernel to exceed shared memory limits.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_shared_memory_exceed():
    # Choose dimensions such that H*W is very large.
    # On many GPUs, the per–block shared memory is limited (~48KB–96KB).
    # Here, using H=W=128 yields 16384 floats ~65KB, which should exceed the limit on many devices.
    N, C, H, W = 1, 1, 128, 128
    x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float32)
    # For this test we'll use dummy weight and bias tensors
    weight = torch.ones(C, device="cuda", dtype=torch.float32)
    bias = torch.zeros(C, device="cuda", dtype=torch.float32)
    kernel = build_kernel()
    with pytest.raises(RuntimeError):
        # Expect the kernel launch to fail due to excessive shared memory usage.
        y = kernel.forward(x, weight, bias, 1e-5)
        torch.cuda.synchronize()

# Issue 4: Test that passing a tensor of type other than float32 triggers issues.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_non_float32():
    N, C, H, W = 4, 4, 32, 32
    # Create a tensor with a type double.
    x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float64)
    weight = torch.ones(C, device="cuda", dtype=torch.float64)
    bias = torch.zeros(C, device="cuda", dtype=torch.float64)
    kernel = build_kernel()
    # The kernel expects float32 pointers; passing double inputs should lead to a crash or error.
    with pytest.raises((RuntimeError, AssertionError)):
        y = kernel.forward(x, weight, bias, 1e-5)
        torch.cuda.synchronize()

# Issue 3: Test on a tensor whose spatial dimensions trigger one of the heuristic cases.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_block_size_heuristic():
    # This test uses a moderately sized input where H*W is small (and the block size heuristic chooses a small block).
    N, C, H, W = 2, 3, 8, 8   # HW = 64, which falls into the first branch (block_size = 128 according to the heuristic)
    x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float32)
    # Using weight and bias provided to trigger the scaling branch.
    weight = torch.randn(C, device="cuda", dtype=torch.float32)
    bias = torch.randn(C, device="cuda", dtype=torch.float32)
    kernel = build_kernel()
    # Run the custom kernel
    y_cuda = kernel.forward(x, weight, bias, 1e-5)
    torch.cuda.synchronize()

    # A reference using PyTorch's InstanceNorm2d
    inorm = torch.nn.InstanceNorm2d(C, eps=1e-5)
    # Set the weight and bias to same values as given (assigning to running parameters)
    inorm.weight.data = weight.clone()
    inorm.bias.data = bias.clone()
    y_ref = inorm(x)

    # The outputs may diverge slightly due to reduction order differences, but they should be close.
    assert torch.allclose(y_cuda, y_ref, atol=1e-4), "Kernel normalization output does not match reference output."

# Issue 1: Although race conditions are notoriously hard to trigger deterministically,
# we can run a repeated execution test on inputs that force the reduction routine to use multiple warps.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_reduction_race_condition():
    N, C, H, W = 1, 16, 16, 16  # 256 elements per channel ensures that multiple warps may be used in reduction.
    x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float32)
    weight = torch.ones(C, device="cuda", dtype=torch.float32)
    bias = torch.zeros(C, device="cuda", dtype=torch.float32)
    kernel = build_kernel()
    # Run the kernel several times in a loop.
    outputs = []
    for _ in range(10):
        y = kernel.forward(x, weight, bias, 1e-5)
        torch.cuda.synchronize()
        outputs.append(y)
    # All outputs should be very close; if a race condition occurs, there might be divergence.
    for i in range(1, len(outputs)):
        assert torch.allclose(outputs[i], outputs[0], atol=1e-4), "Inconsistent results detected, possible race condition in block reduction!"
