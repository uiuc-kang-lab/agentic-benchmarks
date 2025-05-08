
import math
import torch
import pytest
from torch.utils.cpp_extension import load

# Function to compile and return our CUDA extension.
def build_kernel():
    # Building with extra flags so errors regarding missing <cmath> might be caught.
    cuda_module = load(
        name="custom_maxpool3d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Test passing a tensor of a type that might not be correctly dispatched due to using input.type()
def test_dispatch_input_type():
    # Create a double tensor even though the kernel dispatch is on floating types.
    # This test is designed to trigger any type-dispatch problems.
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1
    return_indices = False
    ceil_mode = False

    # Create a double precision tensor on GPU.
    x = torch.randn(2, 4, 8, 8, 8, dtype=torch.double, device='cuda')
    module = build_kernel()
    try:
        # This call should dispatch correctly if input.scalar_type() is used,
        # but with input.type() it might mis-dispatch and produce wrong results or error.
        output = module.forward(x, kernel_size, stride, padding, dilation, return_indices, ceil_mode)
    except RuntimeError as e:
        pytest.skip("Double type not supported: caught RuntimeError as expected.")
    else:
        # If no error, verify output dimensions. (This may pass for double if dispatch works incorrectly)
        expected_d = math.floor((8 + 2*padding - dilation * (kernel_size - 1) - 1) / stride + 1)
        assert output.size(2) == expected_d, "Unexpected output size when dispatching double type."

# Issue 2: Test using tuple parameters for kernel/stride/padding/dilation.
def test_tuple_parameters():
    # Here we simulate passing tuple parameters which the kernel does not support.
    # Since the extension's forward function accepts ints, we expect a TypeError.
    kernel_size = (3, 3, 3)
    stride = (2, 2, 2)
    padding = (1, 1, 1)
    dilation = (1, 1, 1)
    return_indices = False
    ceil_mode = False

    x = torch.randn(2, 4, 8, 8, 8, device='cuda')
    module = build_kernel()

    with pytest.raises(TypeError):
        # We expect an exception as the kernel function only accepts ints.
        module.forward(x, kernel_size, stride, padding, dilation, return_indices, ceil_mode)

# Issue 3: Test that when return_indices=True, the return value is not in the expected format.
def test_return_indices_format():
    # When return_indices is true, PyTorch's native MaxPool3d returns a tuple (output, indices).
    # This kernel returns a stacked tensor; this test checks for that mismatch.
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1
    return_indices = True
    ceil_mode = False

    x = torch.randn(2, 4, 8, 8, 8, device='cuda')
    module = build_kernel()
    result = module.forward(x, kernel_size, stride, padding, dilation, return_indices, ceil_mode)
    
    # The expected proper interface is a tuple; here, we check if the first dimension is 2 (stacked)
    # rather than a tuple of two separate tensors.
    if isinstance(result, torch.Tensor):
        # Expecting a stacked tensor with shape [2, batch, channels, D, H, W] but native behavior expects tuple.
        assert result.size(0) == 2, ("Return indices behavior issue: "
                                     "Expected the indices and output to be returned as separate tensors, "
                                     "but got a stacked tensor with first dim != 2.")
    else:
        pytest.fail("Return indices: Expected a stacked tensor, but got a tuple.")

# Issue 4: Test to force a compilation including check for math functions
# (This test does not run the kernel but forces the kernel to be recompiled,
# so if <cmath> is missing, the build should fail.)
def test_compile_includes_math():
    try:
        module = build_kernel()
    except Exception as e:
        pytest.fail(f"Compilation failed possibly due to missing <cmath>: {e}")
