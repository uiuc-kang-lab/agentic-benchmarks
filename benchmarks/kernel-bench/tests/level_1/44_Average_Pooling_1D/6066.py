
import torch
import pytest
from torch.utils.cpp_extension import load
import torch.nn as nn

def build_kernel():
    cuda_module = load(
        name="avg_pool1d_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Helper: reference average pooling using PyTorch's built-in layer.
def ref_avg_pool1d(x, kernel_size, stride, padding):
    pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=padding)
    return pool(x)

# Test case 1: Trigger shared memory tile size mis‐calculation.
# Use parameters that result in boundary conditions where the needed shared memory tile is unlikely
# to match the statically computed allocation.
def test_shared_memory_issue():
    # Parameters chosen so that the computed global_in_start and global_in_end
    # are not the maximum possible tile size assumed in the kernel.
    batch_size = 4
    in_channels = 3
    input_length = 37  # small number to force boundaries
    kernel_size = 5
    stride = 3
    padding = 2

    # Create an input tensor on CUDA.
    x = torch.randn(batch_size, in_channels, input_length, device="cuda", dtype=torch.float32)

    my_module = build_kernel()
    # Call the custom CUDA kernel
    y_kernel = my_module.forward(x, kernel_size, stride, padding)
    # Call PyTorch's built-in layer
    y_ref = ref_avg_pool1d(x, kernel_size, stride, padding)

    # In case of shared memory mis-allocation the results may be incorrect.
    # Here we check that the outputs differ beyond the tolerance.
    if torch.allclose(y_kernel, y_ref, atol=1e-5):
        pytest.fail("Shared memory tile miscalculation issue was not triggered: outputs match reference despite expected bug.")

# Test case 2: Trigger the lack of support for data types other than float32.
def test_dtype_issue():
    batch_size = 4
    in_channels = 3
    input_length = 64
    kernel_size = 3
    stride = 1
    padding = 1

    # Create an input tensor of type double.
    x = torch.randn(batch_size, in_channels, input_length, device="cuda", dtype=torch.double)

    my_module = build_kernel()
    # Expect the kernel to check for float32 and throw an error.
    with pytest.raises(RuntimeError):
        # This should trigger a TORCH_CHECK failure from the kernel.
        _ = my_module.forward(x, kernel_size, stride, padding)

# Test case 3: Trigger potential incorrect results with non-contiguous inputs.
def test_noncontiguous_issue():
    batch_size = 4
    in_channels = 3
    input_length = 128
    kernel_size = 4
    stride = 2
    padding = 1

    # Create a contiguous input first.
    x_contig = torch.randn(batch_size, in_channels, input_length, device="cuda", dtype=torch.float32)
    # Make a non-contiguous version by transposing and then transposing back.
    # This breaks the assumption on contiguous layout even though the shape is restored.
    x_noncontig = x_contig.transpose(1, 2).transpose(1, 2)
    assert not x_noncontig.is_contiguous(), "The input should be non-contiguous to trigger the issue."

    my_module = build_kernel()
    y_kernel = my_module.forward(x_noncontig, kernel_size, stride, padding)
    y_ref = ref_avg_pool1d(x_noncontig.contiguous(), kernel_size, stride, padding)
    # With non-contiguous input the kernel’s plain pointer arithmetic may produce wrong results.
    if torch.allclose(y_kernel, y_ref, atol=1e-4):
        pytest.fail("Non-contiguous input issue was not triggered: outputs match reference despite expected bug.")

if __name__ == '__main__':
    pytest.main([__file__])
