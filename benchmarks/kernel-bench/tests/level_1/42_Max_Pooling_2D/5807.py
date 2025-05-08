
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="max_pool2d_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1: Using non-floating point tensor (e.g., int tensor) should trigger incorrect initialization behavior.
def test_non_floating_point_input():
    # Create an integer tensor input (this is an incorrect type for max pooling in our kernel).
    batch_size, channels, height, width = 2, 3, 8, 8
    int_input = torch.randint(0, 10, (batch_size, channels, height, width), device="cuda", dtype=torch.int32)
    module = build_kernel()
    
    # Expect the function to fail or produce incorrect results due to the use of -infinity initialization.
    with pytest.raises(RuntimeError):
        output = module.forward(int_input, 2, 2, 1, 1)
        torch.cuda.synchronize()

# Issue 2: Grid dimension limitation (batch_size * channels exceeds CUDA grid z-limit)
def test_exceeding_grid_dimension():
    # Pick dimensions such that batch_size*channels > 65535.
    # For example, setting channels=70000 and batch_size=1.
    batch_size = 1
    channels = 70000
    height = 16
    width = 16
    # Input is a floating point tensor, which is allowed.
    input_tensor = torch.randn(batch_size, channels, height, width, device="cuda", dtype=torch.float32)
    module = build_kernel()
    
    # This call likely triggers a grid configuration issue.
    # We expect either a launch failure or wrong behavior (here we check for an exception).
    with pytest.raises(RuntimeError):
        output = module.forward(input_tensor, 2, 2, 1, 1)
        torch.cuda.synchronize()

# Issue 3: Unqualified use of max function (device usage of max vs. fmax)
def test_comparison_behavior():
    # Although the error might not be obvious, we create an input that exercises
    # the comparison operation in the kernel. Here we use a controlled input where
    # the correct max pooling result is known.
    batch_size, channels, height, width = 1, 1, 4, 4
    input_tensor = torch.tensor([[[[1.0, 2.0, 3.0, 4.0],
                                   [5.0, 6.0, 7.0, 8.0],
                                   [9.0, 10.0, 11.0, 12.0],
                                   [13.0, 14.0, 15.0, 16.0]]]], device="cuda", dtype=torch.float32)
    module = build_kernel()
    
    # We use kernel_size=2, stride=2, padding=0, dilation=1.
    # Expected output: [[[[6.0, 8.0], [14.0, 16.0]]]]
    output = module.forward(input_tensor, 2, 2, 0, 1)
    torch.cuda.synchronize()
    expected = torch.tensor([[[[6.0, 8.0],
                               [14.0, 16.0]]]], device="cuda", dtype=torch.float32)
    # If the unqualified max leads to a wrong behavior, the tolerance assertion might fail.
    assert torch.allclose(output, expected, atol=1e-5), "Kernel max operation produced incorrect results."

# Issue 4: Non-contiguous input tensor handling.
def test_non_contiguous_input():
    # Create a contiguous input and then make it non-contiguous by transposing
    batch_size, channels, height, width = 2, 3, 8, 8
    input_tensor = torch.randn(batch_size, channels, height, width, device="cuda", dtype=torch.float32)
    non_contiguous_input = input_tensor.transpose(2, 3)  # Now non-contiguous.
    module = build_kernel()
    
    # The kernel assumes contiguous memory; this test should expose potential mis-indexing.
    with pytest.raises(RuntimeError):
        output = module.forward(non_contiguous_input, 2, 2, 1, 1)
        torch.cuda.synchronize()
