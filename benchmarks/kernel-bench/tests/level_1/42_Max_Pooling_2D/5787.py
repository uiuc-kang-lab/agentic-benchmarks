
import pytest
import torch
from torch.utils.cpp_extension import load

# Build the CUDA extension from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Unused shared-memory allocation.
# While this does not affect correctness, we can expose potential problems by using input sizes
# that are not multiples of the tile dimensions, ensuring that threads near boundaries execute.
def test_non_divisible_tile():
    # Using kernel parameters that yield an output shape not a multiple of the tile size (8x8)
    batch_size = 2
    channels = 2
    height = 17  # deliberately chosen
    width = 17
    kernel_size = 3  # not equal to 2 => uses generic loop branch
    stride = 2
    padding = 1
    dilation = 1

    input_tensor = torch.randn(batch_size, channels, height, width, device="cuda", dtype=torch.float32)
    # Compute expected output using PyTorch's built in function.
    expected = torch.nn.functional.max_pool2d(
        input_tensor, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation
    )
    module = build_kernel()
    output = module.forward(input_tensor, kernel_size, stride, padding, dilation)
    torch.cuda.synchronize()
    # Compare against PyTorch if sizes are as expected.
    assert torch.allclose(output, expected, atol=1e-4), "Output does not match expected result for non-divisible tile sizes."

# Issue 2: Improper use of the max operation.
# This can be exposed by using double precision where differences
# in max functions (like fmax versus operator overloads) may produce discrepancies.
def test_double_precision():
    batch_size = 1
    channels = 1
    height = 16
    width = 16
    kernel_size = 2  # will exercise the unrolled loop branch
    stride = 2
    padding = 0
    dilation = 1

    # Use double precision
    input_tensor = torch.randn(batch_size, channels, height, width, device="cuda", dtype=torch.float64)
    expected = torch.nn.functional.max_pool2d(
        input_tensor, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation
    )
    module = build_kernel()
    output = module.forward(input_tensor, kernel_size, stride, padding, dilation)
    torch.cuda.synchronize()
    # There may be issues with the max operation in double; test for discrepancies.
    assert torch.allclose(output, expected, atol=1e-5), "Double precision max operation produced unexpected results."

# Issue 3: Inappropriate initialization for non-floating types.
# Passing an integer tensor should trigger an error due to the use of -infinity.
def test_integer_input():
    batch_size = 1
    channels = 1
    height = 16
    width = 16
    kernel_size = 2  
    stride = 2
    padding = 0
    dilation = 1

    input_tensor = torch.randint(0, 100, (batch_size, channels, height, width), device="cuda", dtype=torch.int32)
    module = build_kernel()
    with pytest.raises(Exception):
        # The kernel is expected to fail because integer types
        # cannot represent -infinity and the max operation is not properly defined.
        _ = module.forward(input_tensor, kernel_size, stride, padding, dilation)
    torch.cuda.synchronize()

# Issue 4: Lack of generality for kernel shape.
# Although the PyTorch layer might expect more flexibility, our kernel only supports a single value for kernel_size.
# We trigger this by testing with a non-2 kernel (other than the unrolled branch) and comparing with PyTorch.
def test_non_square_kernel_emulation():
    # Here, we emulate a more general case by using a kernel_size value different from 2.
    batch_size = 2
    channels = 3
    height = 20
    width = 20
    kernel_size = 3  # this forces the generic branch
    stride = 1
    padding = 1
    dilation = 2  # increase dilation to further test the kernel arithmetic

    input_tensor = torch.randn(batch_size, channels, height, width, device="cuda", dtype=torch.float32)
    expected = torch.nn.functional.max_pool2d(
        input_tensor, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation
    )
    module = build_kernel()
    output = module.forward(input_tensor, kernel_size, stride, padding, dilation)
    torch.cuda.synchronize()
    assert torch.allclose(output, expected, atol=1e-4), "Generic kernel branch produced incorrect results for non-square kernel parameters."

if __name__ == "__main__":
    pytest.main([__file__])
