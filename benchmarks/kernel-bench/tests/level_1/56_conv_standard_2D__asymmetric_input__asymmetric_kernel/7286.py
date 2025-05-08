
import torch
import torch.nn.functional as F
import pytest
from torch.utils.cpp_extension import load
import math

def build_kernel():
    cuda_module = load(
        name="custom_conv_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Helper function to compute convolution using PyTorch's native conv2d for reference.
def ref_conv2d(input, weight, bias, stride, padding, dilation, groups):
    return F.conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

@pytest.fixture(scope="module")
def cuda_kernel_module():
    module = build_kernel()
    return module

# Issue 1 test: Incorrect shared memory usage (tile reuse)
# Even though the kernel may compute some result, the non-tiled memory loading may result in numerical inaccuracies
# on boundary conditions. We try a scenario with a moderately large kernel and asymmetric sizes.
def test_shared_memory_usage(cuda_kernel_module):
    # Configuration with non-square kernel and input to stress data reuse
    batch_size = 2
    in_channels = 3
    out_channels = 4
    kernel_size = (5, 7)  # larger kernel to force multiple iterations in loops
    stride = (1, 1)
    padding = (2, 3)
    dilation = (1, 1)
    groups = 1

    # Input dimensions where image is moderately sized.
    H_in = 32
    W_in = 48

    # Create random input and weight tensors
    x = torch.randn(batch_size, in_channels, H_in, W_in, device="cuda", dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1], device="cuda", dtype=torch.float32)
    bias = None  # No bias

    # Get kernel output
    out_kernel = cuda_kernel_module.forward(x, weight, bias, list(stride), list(padding), list(dilation), groups)
    # Get reference output
    out_ref = ref_conv2d(x, weight, bias, stride, padding, dilation, groups)

    # Compare results (they may differ due to the inefficient memory usage if there is a bug)
    max_diff = (out_kernel - out_ref).abs().max().item()
    assert math.isclose(max_diff, 0., abs_tol=1e-3), f"Issue with shared memory usage: max difference {max_diff} exceeds tolerance"

# Issue 2 test: Excessive __syncthreads calls may impact performance and even cause deadlocks in more complex kernels.
# While performance is hard to test with a unit test, we can create a scenario with an unusually large kernel
# (forcing many inner loop iterations) and check for timeouts or hangs.
def test_excessive_syncthreads(cuda_kernel_module):
    batch_size = 1
    in_channels = 3
    out_channels = 3
    kernel_size = (11, 11)  # Very large kernel to cause many iterations and synchronizations
    stride = (1, 1)
    padding = (5, 5)
    dilation = (1, 1)
    groups = 1

    H_in = 64
    W_in = 64

    x = torch.randn(batch_size, in_channels, H_in, W_in, device="cuda", dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1], device="cuda", dtype=torch.float32)
    bias = None

    # Run kernel and ensure it completes (if __syncthreads is misused, it might hang)
    out_kernel = cuda_kernel_module.forward(x, weight, bias, list(stride), list(padding), list(dilation), groups)
    torch.cuda.synchronize()  # force synchronization; if kernel hangs, the test will timeout

    # Check vs reference result for sanity
    out_ref = ref_conv2d(x, weight, bias, stride, padding, dilation, groups)
    max_diff = (out_kernel - out_ref).abs().max().item()
    assert math.isclose(max_diff, 0., abs_tol=1e-3), f"Excessive synchronization issue: max diff {max_diff}"

# Issue 3 test: Hardcoded shared memory size
# Create a scenario where the number of threads per block might exceed the allocated shared memory size.
# Since the kernel hardcodes dynamic shared memory size (1024 floats) and uses threadIdx.x indexing,
# we simulate this by forcing a large output spatial size so that the kernel might be launched with a block size
# that – if the code were to change to use a variable block size – would overflow the shared memory allocated.
def test_shared_memory_allocation(cuda_kernel_module):
    batch_size = 1
    in_channels = 1
    out_channels = 1
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = (1, 1)
    dilation = (1, 1)
    groups = 1

    # Choose an input so that output spatial size is large.
    H_in = 128
    W_in = 128

    x = torch.randn(batch_size, in_channels, H_in, W_in, device="cuda", dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1], device="cuda", dtype=torch.float32)
    bias = None

    # The kernel launch configuration in the kernel code uses threads_per_block = 256.
    # Here, we do not control block size directly, but if someone attempted to change it without updating the shared memory allocation,
    # out-of-bound access might occur.
    #
    # We run the kernel and check that it completes without CUDA errors. (The test will fail if there was an out-of-bounds access.)
    out_kernel = cuda_kernel_module.forward(x, weight, bias, list(stride), list(padding), list(dilation), groups)
    torch.cuda.synchronize()  # if shared memory indexing is out of bounds, error should be reported.

    # For sanity compare with reference convolution
    out_ref = ref_conv2d(x, weight, bias, stride, padding, dilation, groups)
    max_diff = (out_kernel - out_ref).abs().max().item()
    assert math.isclose(max_diff, 0., abs_tol=1e-3), f"Shared memory allocation issue: max diff {max_diff} exceeds tolerance"
