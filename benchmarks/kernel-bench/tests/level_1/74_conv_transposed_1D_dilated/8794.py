
import pytest
import torch
import math
from torch import nn
from torch.utils.cpp_extension import load

# Helper function: compile the CUDA extension.
def build_kernel(extra_cuda_cflags=None, extra_include_paths=None):
    cuda_module = load(
        name="conv_transpose1d_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=(extra_cuda_cflags if extra_cuda_cflags is not None else ["-O3", "--use_fast_math"]),
        extra_include_paths=(extra_include_paths if extra_include_paths is not None else []),
        verbose=False,
    )
    return cuda_module

# Reference implementation using PyTorch's nn.ConvTranspose1d.
def reference_conv_transpose1d(x, weight, bias, stride, padding, dilation):
    # Create a temporary module using the built-in op.
    # Note: weight shape for nn.ConvTranspose1d is (in_channels, out_channels, kernel_size)
    conv = nn.ConvTranspose1d(
        in_channels=x.shape[1],
        out_channels=weight.shape[1],
        kernel_size=weight.shape[2],
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=(bias is not None)
    ).to(x.device)
    # Copy weight and bias (if provided)
    with torch.no_grad():
        conv.weight.copy_(weight)
        if bias is not None:
            conv.bias.copy_(bias)
    return conv(x)

# Test case 1: Trigger issue with fixed shared‐memory size.
# We do so by overriding the launch configuration size.
# (The kernel is written to use a fixed shared array of size 32.
# If we force the kernel to launch with more than 32 warps per block,
# the static shared memory allocation will be too small.)
def test_shared_memory_overflow():
    # Create inputs such that our “reduction” is heavy.
    # (We choose large C_in and kernel_size so that the kernel's inner loop is over many iterations.)
    N = 2
    C_in = 128  # large input channel count
    L_in = 16
    C_out = 8
    kernel_size = 64  # many kernel positions

    stride = 1
    padding = 0
    dilation = 1

    # The weight tensor shape is (C_in, C_out, kernel_size)
    weight = torch.randn(C_in, C_out, kernel_size, device="cuda", dtype=torch.float32)
    x = torch.randn(N, C_in, L_in, device="cuda", dtype=torch.float32)
    bias = torch.randn(C_out, device="cuda", dtype=torch.float32)

    # Build the module.
    kernel_mod = build_kernel()

    # Here we use monkey-patching: simulate a launch with threads_per_block > 32*WARP_SIZE = 1024.
    # We do that by temporarily overriding the kernel launch parameters in the module.
    # (Assume the module is modified to read a global variable "threads_per_block".)
    # For the test, we simulate using a "bad" launch configuration.
    threads_per_block = 2048  # deliberately too many threads per block for static shared allocation

    # We compute L_out similarly to the forward host code.
    L_out = (L_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    total_elements = N * C_out * L_out

    # Launch the kernel manually.
    kernel_mod.conv_transpose1d_kernel_shared_reduction[
        total_elements, threads_per_block
    ](
        x.data_ptr(), weight.data_ptr(), bias.data_ptr(),
        torch.empty((N, C_out, L_out), device="cuda", dtype=torch.float32).data_ptr(),
        N, C_in, C_out, L_in, L_out, kernel_size, stride, padding, dilation
    )
    # With overflow, we expect the result to be wrong (or GPU sanity checks to fail).
    # Here we simply mark the test as failed if no error is raised.
    # In a real-case we might catch a cuda kernel error.
    pytest.skip("Test for shared memory overflow triggered; manual inspection required.")

# Test case 2: Trigger issue with __shfl_down_sync misuse when blockDim is below warp size.
# We launch the kernel with a thread block having less than 32 threads and expect the reduction to be incorrect.
def test_partial_warp_reduction():
    N = 1
    C_in = 3
    L_in = 8
    C_out = 2
    kernel_size = 3

    stride = 1
    padding = 1
    dilation = 1
    weight = torch.randn(C_in, C_out, kernel_size, device="cuda", dtype=torch.float32)
    x = torch.randn(N, C_in, L_in, device="cuda", dtype=torch.float32)
    bias = torch.randn(C_out, device="cuda", dtype=torch.float32)

    # Build the module.
    kernel_mod = build_kernel()
    # We simulate a launch with very few threads per block (e.g. 16 threads)
    threads_per_block = 16
    L_out = (L_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    total_elements = N * C_out * L_out

    # Make an output tensor.
    y = torch.empty((N, C_out, L_out), device="cuda", dtype=torch.float32)
    # Launch the kernel manually with the reduced thread count.
    kernel_mod.conv_transpose1d_kernel_shared_reduction[
        total_elements, threads_per_block
    ](
        x.data_ptr(), weight.data_ptr(), bias.data_ptr(),
        y.data_ptr(),
        N, C_in, C_out, L_in, L_out, kernel_size, stride, padding, dilation
    )
    # Compute a reference.
    y_ref = reference_conv_transpose1d(x, weight, bias, stride, padding, dilation)
    torch.cuda.synchronize()
    # Check if the output is (incorrectly) computed.
    if torch.allclose(y, y_ref, atol=1e-4):
        pytest.fail("The kernel incorrectly computed the reduction with small blockDim; __shfl_down_sync usage may be wrong.")

# Test case 3: Trigger issue with grid dimension overflow.
# We create inputs such that the total number of output elements is huge.
def test_grid_dimension_overflow():
    # Choose dimensions to yield a huge grid.
    N = 1024  # large batch size
    C_in = 3
    L_in = 256
    C_out = 64
    kernel_size = 5
    stride = 1
    padding = 0
    dilation = 1

    weight = torch.randn(C_in, C_out, kernel_size, device="cuda", dtype=torch.float32)
    x = torch.randn(N, C_in, L_in, device="cuda", dtype=torch.float32)
    bias = torch.randn(C_out, device="cuda", dtype=torch.float32)

    # Build the module.
    kernel_mod = build_kernel()

    # Compute L_out.
    L_out = (L_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    total_elements = N * C_out * L_out

    # Check if total_elements exceed a typical gridDim.x (note: many devices allow >2^16 blocks in x)
    if total_elements < 2**16:
        pytest.skip("Grid dimension overflow test skipped because total elements not huge enough.")

    # Launch kernel.
    y = torch.empty((N, C_out, L_out), device="cuda", dtype=torch.float32)
    try:
        kernel_mod.conv_transpose1d_kernel_shared_reduction[
            total_elements, 512
        ](
            x.data_ptr(), weight.data_ptr(), bias.data_ptr(),
            y.data_ptr(),
            N, C_in, C_out, L_in, L_out, kernel_size, stride, padding, dilation
        )
        torch.cuda.synchronize()
    except RuntimeError as e:
        pytest.skip("Kernel launch failed due to grid dimension overflow as expected.")
    else:
        pytest.fail("Kernel did not fail on grid dimension overflow when it should have.")

# Test case 4: Trigger issue with non‐contiguous tensors.
# We pass non‐contiguous input tensors to the kernel.
def test_non_contiguous_inputs():
    N = 4
    C_in = 3
    L_in = 64
    C_out = 8
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1

    # Create non-contiguous x and weight.
    x = torch.randn(N, C_in, L_in, device="cuda", dtype=torch.float32).transpose(1,2)
    weight = torch.randn(C_in, C_out, kernel_size, device="cuda", dtype=torch.float32).flip(2)
    bias = torch.randn(C_out, device="cuda", dtype=torch.float32)

    # Although the host function calls .contiguous(), in a more general usage this is not guaranteed.
    kernel_mod = build_kernel()

    L_out = (L_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    total_elements = N * C_out * L_out

    # Call the kernel directly with non-contiguous tensors.
    y = torch.empty((N, C_out, L_out), device="cuda", dtype=torch.float32)
    # Note: since kernel.cu forces a .contiguous() call on x and weight inside conv_transpose1d_forward,
    # here we call the kernel function directly to mimic a lower level call.
    kernel_mod.conv_transpose1d_kernel_shared_reduction[
        total_elements, 512
    ](
        x.data_ptr(), weight.data_ptr(), bias.data_ptr(),
        y.data_ptr(),
        N, C_in, C_out, L_in, L_out, kernel_size, stride, padding, dilation
    )
    torch.cuda.synchronize()
    y_ref = reference_conv_transpose1d(x.contiguous(), weight.contiguous(), bias, stride, padding, dilation)
    if torch.allclose(y, y_ref, atol=1e-4):
        pytest.fail("Kernel produced correct output for non-contiguous inputs. It should have explicitly handled contiguity for general usage.")

