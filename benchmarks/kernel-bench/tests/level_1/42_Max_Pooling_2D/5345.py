
import torch
import pytest
from torch.utils.cpp_extension import load
import torch.nn as nn

# Helper function to compile and load the CUDA kernel
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# A helper function to run the custom CUDA forward and compare with PyTorch's MaxPool2d
def run_and_compare(input_tensor, kernel_size, stride, padding, dilation):
    cuda_mod = build_kernel()
    # Call our custom CUDA kernel
    out_cuda = cuda_mod.forward(input_tensor, kernel_size, stride, padding, dilation)
    # Use PyTorch's native implementation for reference.
    maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    out_ref = maxpool(input_tensor)
    torch.cuda.synchronize()
    # We expect the outputs to be very close.
    assert torch.allclose(out_cuda, out_ref, atol=1e-4), (
        f"Difference detected for kernel_size={kernel_size}, stride={stride}, "
        f"padding={padding}, dilation={dilation}. Max diff: {(out_cuda - out_ref).abs().max().item()}"
    )

# Issue 1: Test for potential incorrect initialization of max value using std::numeric_limits<scalar_t>::infinity()
# Use a case with many negative values to see if the kernel incorrectly initializes max.
def test_incorrect_infinity():
    batch_size, channels, height, width = 4, 3, 16, 16
    # Make input with predominantly negative numbers.
    input_tensor = torch.randn(batch_size, channels, height, width, device="cuda") - 10.0
    # Use kernel_size 2 (which follows the first branch that uses std::numeric_limits<scalar_t>::infinity())
    run_and_compare(input_tensor, kernel_size=2, stride=2, padding=1, dilation=1)

# Issue 2: Test for ambiguity/inconsistency in using "max" without proper qualification.
# We use double precision input to possibly expose problems with function overloading.
def test_unqualified_max_double():
    batch_size, channels, height, width = 4, 3, 16, 16
    # Create input tensor in double precision.
    input_tensor = torch.randn(batch_size, channels, height, width, device="cuda", dtype=torch.float64)
    run_and_compare(input_tensor, kernel_size=2, stride=2, padding=1, dilation=1)

# Issue 3: Test concurrent kernel launches to stress constant memory usage.
# Running several kernels simultaneously (in separate streams) using different parameters.
def test_constant_memory_collision():
    batch_size, channels, height, width = 4, 3, 32, 32
    input_tensor1 = torch.randn(batch_size, channels, height, width, device="cuda")
    input_tensor2 = torch.randn(batch_size, channels, height, width, device="cuda")
    
    # Create two streams
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()

    cuda_mod = build_kernel()

    # Launch first kernel on stream1 with kernel_size 2
    with torch.cuda.stream(stream1):
        out1 = cuda_mod.forward(input_tensor1, 2, 2, 1, 1)
    # Launch second kernel on stream2 with kernel_size 3
    with torch.cuda.stream(stream2):
        out2 = cuda_mod.forward(input_tensor2, 3, 2, 1, 1)
    
    # Synchronize streams
    stream1.synchronize()
    stream2.synchronize()

    # Compare results with native PyTorch layers
    maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1, dilation=1)
    maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1)
    ref1 = maxpool1(input_tensor1)
    ref2 = maxpool2(input_tensor2)
    
    assert torch.allclose(out1, ref1, atol=1e-4), "Concurrent kernel launch for kernel_size=2 produced incorrect results."
    assert torch.allclose(out2, ref2, atol=1e-4), "Concurrent kernel launch for kernel_size=3 produced incorrect results."

# Issue 4: Test with a kernel_size that is not 2 or 3 (e.g. 4) to force execution of the non-unrolled branch,
# which might reveal performance or correctness problems in more general situations.
def test_non_unrolled_kernel_size():
    batch_size, channels, height, width = 4, 3, 32, 32
    # Using kernel_size 4; this will use the else branch (template parameter -1) with runtime loops.
    input_tensor = torch.randn(batch_size, channels, height, width, device="cuda")
    run_and_compare(input_tensor, kernel_size=4, stride=2, padding=1, dilation=1)
