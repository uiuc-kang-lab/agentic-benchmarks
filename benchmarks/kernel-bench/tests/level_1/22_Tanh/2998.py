
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the CUDA extension from the provided kernel.cu file.
    cuda_module = load(
        name="tanh_ext",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Non-contiguous input for float32.
# Expectation: Due to misaligned data (non-contiguous view), the output may be incorrect.
def test_non_contiguous_float():
    my_module = build_kernel()
    # Create a contiguous tensor then a transposed copy to force non-contiguity.
    x = torch.randn(32, 64, device="cuda", dtype=torch.float32)
    x_noncontig = x.t()  # This is non-contiguous.
    # Run our custom CUDA kernel.
    out = my_module.forward(x_noncontig)
    expected = torch.tanh(x_noncontig)
    # The outputs are expected to differ if the kernel incorrectly assumes contiguity.
    # We check if they do NOT match.
    if torch.allclose(out, expected, atol=1e-5):
        pytest.fail("Kernel returned correct results on non-contiguous input, but it should have failed due to assumed alignment issues.")

# Test 2: Input tensor with total elements not divisible by 4.
# Expectation: Although the kernel has a tail kernel, if a tensor has an
# unaligned starting address (e.g., from a slice) the bulk float4 access may be unsafe.
def test_inexact_divisible_elements():
    my_module = build_kernel()
    # Create a contiguous tensor of 103 elements (not divisible by 4) using 1D view.
    x = torch.randn(103, device="cuda", dtype=torch.float32)
    out = my_module.forward(x)
    expected = torch.tanh(x)
    # If the tail logic misbehaves due to alignment issues, the outputs will differ.
    if torch.allclose(out, expected, atol=1e-5):
        pytest.fail("Kernel returned correct results on inexact divisible elements input, but an alignment issue is expected.")

# Test 3: Using a type other than float32 (e.g., int32) to trigger unsupported type behavior.
# Expectation: The kernel dispatch for non-float types is limited to floating types.
def test_wrong_input_type_int():
    my_module = build_kernel()
    x = torch.randint(low=-10, high=10, size=(1024,), device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError):
        # The AT_DISPATCH_FLOATING_TYPES macro is not designed for int types,
        # so we expect the extension to raise an error.
        my_module.forward(x)

# Test 4: Using a half (float16) input, which is not specially handled by the vectorized float4 kernels.
# Expectation: The generic kernel dispatch for half is not provided by AT_DISPATCH_FLOATING_TYPES
# in our code and may produce a wrong result or error.
def test_half_precision_input():
    my_module = build_kernel()
    x = torch.randn(1024, device="cuda", dtype=torch.float16)
    try:
        out = my_module.forward(x)
        expected = torch.tanh(x)
        # If the kernel executes without error, verify that the results are close.
        # (They might be different because the kernel may be using the wrong math function.)
        if not torch.allclose(out, expected, atol=1e-3):
            pytest.fail("Kernel produced incorrect results for half precision input.")
    except RuntimeError as e:
        # If a runtime error is thrown rather than a numerical error, that is acceptable.
        pass
