
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# Utility function to compile the CUDA kernel extension.
def build_kernel():
    cuda_module = load(
        name="pool_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Ambiguous max()-function on device.
# We try to compile and run the kernel with a standard floating input.
# If the wrong max is chosen, the kernel might silently produce wrong output.
# In this test, we compare against PyTorch built-in max_pool2d.
def test_ambiguous_max():
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    batch = 4
    channels = 3
    height = 10
    width = 10

    input_tensor = torch.randn(batch, channels, height, width, device="cuda", dtype=torch.float32)
    torch_out = F.max_pool2d(input_tensor, kernel_size, stride=stride, padding=padding, dilation=dilation)

    pool_kernel = build_kernel()
    # run our custom CUDA forward
    kernel_out = pool_kernel.forward(input_tensor, kernel_size, stride, padding, dilation)
    torch.cuda.synchronize()

    # This test will fail (differences detected) if our ambiguous max call caused issues.
    assert torch.allclose(kernel_out, torch_out, atol=1e-5), \
           f"Ambiguous max device call issue: output mismatch\nCustom: {kernel_out}\nReference: {torch_out}"

# Issue 2: Hard‐coded unrolling for kernel sizes that are not 2 or 3.
# Here, we choose kernel_size = 5 (or another number not handled explicitly).
def test_unroll_kernel_size():
    kernel_size = 5  # not among {2,3}
    stride = 1
    padding = 2
    dilation = 1
    batch = 2
    channels = 4
    height = 16
    width = 16

    input_tensor = torch.randn(batch, channels, height, width, device="cuda", dtype=torch.float32)
    torch_out = F.max_pool2d(input_tensor, kernel_size, stride=stride, padding=padding, dilation=dilation)

    pool_kernel = build_kernel()
    kernel_out = pool_kernel.forward(input_tensor, kernel_size, stride, padding, dilation)
    torch.cuda.synchronize()

    assert torch.allclose(kernel_out, torch_out, atol=1e-5), \
           f"Hard-coded unrolling issue: output mismatch for kernel_size 5."

# Issue 3: Lack of error checking after kernel launch.
# We purposefully provide an invalid configuration so that the kernel should error out.
# For example, choose parameters that result in a negative output shape.
def test_kernel_launch_error():
    kernel_size = 3
    stride = 1
    padding = -5   # invalid negative padding
    dilation = 1
    batch = 1
    channels = 1
    height = 8
    width = 8

    input_tensor = torch.randn(batch, channels, height, width, device="cuda", dtype=torch.float32)
    pool_kernel = build_kernel()

    with pytest.raises(RuntimeError):
        # This should error out on kernel launch / memory access because padding < 0 is not sensible.
        _ = pool_kernel.forward(input_tensor, kernel_size, stride, padding, dilation)
        torch.cuda.synchronize()

# Issue 4: Limited type support.
# Here we try to call the kernel with an integer tensor. Since the extension
# only dispatches over floating types, the call should raise an exception.
def test_integer_type_not_supported():
    kernel_size = 2
    stride = 2
    padding = 0
    dilation = 1
    batch = 2
    channels = 3
    height = 8
    width = 8

    # Create an integer tensor.
    input_tensor = torch.randint(0, 10, (batch, channels, height, width), device="cuda", dtype=torch.int32)
    pool_kernel = build_kernel()

    with pytest.raises(RuntimeError):
        _ = pool_kernel.forward(input_tensor, kernel_size, stride, padding, dilation)
        torch.cuda.synchronize()

# Issue 5: Incompatibility with arbitrary kernel sizes.
# Instead of one of the hard-coded ones (2 or 3), we choose a kernel size of 1.
def test_kernel_size_one():
    kernel_size = 1  # usually a no-op but still should work correctly
    stride = 1
    padding = 0
    dilation = 1
    batch = 3
    channels = 3
    height = 12
    width = 12

    input_tensor = torch.randn(batch, channels, height, width, device="cuda", dtype=torch.float32)
    torch_out = F.max_pool2d(input_tensor, kernel_size, stride=stride, padding=padding, dilation=dilation)

    pool_kernel = build_kernel()
    kernel_out = pool_kernel.forward(input_tensor, kernel_size, stride, padding, dilation)
    torch.cuda.synchronize()

    assert torch.allclose(kernel_out, torch_out, atol=1e-5), \
           f"Arbitrary kernel size issue: output mismatch for kernel_size 1."


if __name__ == '__main__':
    pytest.main([__file__])
