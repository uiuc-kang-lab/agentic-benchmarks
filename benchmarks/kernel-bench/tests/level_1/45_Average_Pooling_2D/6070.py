
import pytest
import torch
from torch.utils.cpp_extension import load

# Utility to build the CUDA kernel extension
def build_kernel():
    cuda_module = load(
        name="pool_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Using half precision input should trigger a dispatch error.
def test_half_precision_not_supported():
    # Build module
    kernel_module = build_kernel()

    # Create a tensor in half precision
    x = torch.randn(2, 3, 32, 32, dtype=torch.half, device="cuda")
    kernel_size = 3
    stride = 3
    padding = 1

    # With AT_DISPATCH_FLOATING_TYPES the kernel does not support half.
    # We expect an error to be raised when the kernel is run.
    with pytest.raises(RuntimeError):
        # Calling the forward function from the kernel module.
        out = kernel_module.forward(x, kernel_size, stride, padding)
        torch.cuda.synchronize()

# Test 2: When padding is nonzero and parts of the pooling region fall outside the input, 
# some frameworks may want to average only over valid (in–bounds) elements (i.e. count_exclude_pad=True).
# Our kernel always divides by kernel_size**2.
def test_count_exclude_pad_issue():
    # Build module
    kernel_module = build_kernel()

    # Create a tensor that when pooling will have some pooling windows with parts outside.
    N, C, H, W = 1, 1, 4, 4
    # Use a tensor with constant value so that we can predict the average.
    x = torch.ones(N, C, H, W, dtype=torch.float32, device="cuda")
    kernel_size = 3
    stride = 1
    padding = 1

    # Forward pass using our custom kernel.
    out_kernel = kernel_module.forward(x, kernel_size, stride, padding)
    torch.cuda.synchronize()

    # Compute a reference “exclude padding” average pooling 
    # manually: For each output element, sum only the valid inputs and divide by the number of valid elements.
    outH = (H + 2 * padding - kernel_size) // stride + 1
    outW = (W + 2 * padding - kernel_size) // stride + 1
    out_ref = torch.empty((N, C, outH, outW), dtype=torch.float32, device="cuda")
    for n in range(N):
        for c in range(C):
            for h_out in range(outH):
                for w_out in range(outW):
                    h_start = h_out * stride - padding
                    w_start = w_out * stride - padding
                    sum_val = 0.0
                    count = 0
                    for i in range(kernel_size):
                        for j in range(kernel_size):
                            h_in = h_start + i
                            w_in = w_start + j
                            if (0 <= h_in < H) and (0 <= w_in < W):
                                sum_val += 1.0  # since value=1.0 everywhere
                                count += 1
                    out_ref[n, c, h_out, w_out] = sum_val / count  # average over valid elements

    # Meanwhile our kernel will always use kernel_size**2 as denominator.
    # For pooling windows that extend outside the image the kernel result will be lower.
    # We expect these results not to match.
    if torch.allclose(out_kernel, out_ref, atol=1e-5):
        raise AssertionError("Kernel averaging incorrectly matches count_exclude_pad behavior.")

# Test 3: Asynchronous kernel errors might be hidden without explicit synchronization.
def test_no_device_synchronize():
    # Build module
    kernel_module = build_kernel()
    
    # Use an invalid kernel configuration that should trigger an error.
    # For instance, by providing a negative stride.
    x = torch.randn(2, 3, 32, 32, dtype=torch.float32, device="cuda")
    kernel_size = 3
    stride = -1  # invalid
    padding = 1

    with pytest.raises(Exception):
        out = kernel_module.forward(x, kernel_size, stride, padding)
        # Attempt to force synchronization in order to raise any asynchronous errors.
        torch.cuda.synchronize()
