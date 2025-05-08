
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

# Test case for Issue 1: Constant memory initialization could lead to wrong parameter values.
# We test by comparing the custom CUDA kernel output with PyTorch's nn.MaxPool2d with nontrivial parameters.
def test_constant_memory_params():
    # Use parameters that would be sensitive if the wrong constant memory entries were used.
    batch_size = 4
    channels = 3
    height = 16
    width = 16
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 2

    x = torch.randn(batch_size, channels, height, width, device="cuda", dtype=torch.float32)
    custom_module = build_kernel()
    # Call the custom CUDA kernel
    out_cuda = custom_module.forward(x, kernel_size, stride, padding, dilation)
    
    # Use PyTorch's MaxPool2d as reference (matches its formula)
    maxpool = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    out_ref = maxpool(x)
    
    torch.cuda.synchronize()
    # If the constant memory had garbage beyond the first four values,
    # the result may differ from the reference.
    assert torch.allclose(out_cuda, out_ref, atol=1e-4), "Output does not match PyTorch reference. Possible constant memory issue."

# Test case for Issue 2: Using an integer type.
# The kernel dispatch macro only handles floating point types.
def test_incorrect_dtype():
    batch_size = 2
    channels = 3
    height = 8
    width = 8
    kernel_size = 2
    stride = 2
    padding = 0
    dilation = 1

    # Create an integer tensor
    x = torch.randint(0, 10, (batch_size, channels, height, width), device="cuda", dtype=torch.int32)
    custom_module = build_kernel()
    with pytest.raises(RuntimeError):
        # This should raise an error because AT_DISPATCH_FLOATING_TYPES won't handle int32.
        custom_module.forward(x, kernel_size, stride, padding, dilation)

# Test case for Issue 3: Lack of error checking for CUDA runtime calls.
# We simulate a situation where the kernel is mis‐invoked. For example, by sending a tensor on CPU.
def test_cuda_runtime_error():
    batch_size = 2
    channels = 3
    height = 8
    width = 8
    kernel_size = 2
    stride = 2
    padding = 0
    dilation = 1

    # Create a tensor on the CPU even though our kernel expects a CUDA tensor.
    x = torch.randn(batch_size, channels, height, width, device="cpu", dtype=torch.float32)
    custom_module = build_kernel()
    with pytest.raises(Exception):
        # This should raise an error from CUDA because the input is not on the GPU.
        custom_module.forward(x, kernel_size, stride, padding, dilation)
    torch.cuda.synchronize()

# Test case for Issue 4: Missing <limits> include.
# This is a compile–time issue. If the module builds successfully, the test passes.
def test_compilation():
    try:
        custom_module = build_kernel()
    except Exception as e:
        pytest.fail(f"Compilation failed, possibly due to missing <limits> include: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
