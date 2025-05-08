
import torch
import pytest
from torch.utils.cpp_extension import load
import os

def build_kernel():
    # Ensure we compile with verbose output to catch any warnings.
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Test with double type input.
def test_double_input():
    # Create a double tensor.
    input_tensor = torch.randn(1024, dtype=torch.double, device="cuda")
    # Reference using torch.relu.
    expected = torch.relu(input_tensor)
    module = build_kernel()

    # The kernel dispatches on AT_DISPATCH_FLOATING_TYPES which includes double,
    # but the vectorized cast to float4 is only valid for float32.
    # Therefore, we expect the output to be different from the expected result.
    output = module.forward(input_tensor)
    torch.cuda.synchronize()
    # Because of wrong reinterpret_cast, the result should differ.
    assert not torch.allclose(output, expected, rtol=1e-5, atol=1e-5), \
        "Expected a mismatch when using double type input due to vectorized cast error."

# Issue 2: Test with a non-contiguous input tensor.
def test_non_contiguous_input():
    # Create a contiguous tensor and then make a non-contiguous slice.
    base = torch.randn(1024, dtype=torch.float32, device="cuda")
    # Create a non-contiguous tensor by transposing a 2D version.
    x = base.view(32, 32).t().reshape(-1)
    assert not x.is_contiguous(), "x should be non-contiguous for this test."
    expected = torch.relu(x)
    module = build_kernel()
    output = module.forward(x)
    torch.cuda.synchronize()
    # In a correct kernel, even if non-contiguous, one would ideally use at::TensorIterator.
    # Here, due to unsafe casting, the output might be wrong.
    assert not torch.allclose(output, expected, rtol=1e-5, atol=1e-5), \
        "Expected a mismatch when using a non-contiguous input due to unhandled alignment issues."

# Issue 3: Test with an input tensor size not divisible by 4.
def test_non_divisible_input_size():
    # Create a tensor with a number of elements not divisible by 4.
    n = 1023  # 1023 is not divisible by 4.
    input_tensor = torch.randn(n, dtype=torch.float32, device="cuda")
    expected = torch.relu(input_tensor)
    module = build_kernel()
    output = module.forward(input_tensor)
    torch.cuda.synchronize()
    # If the kernel’s handling of the remainder is incorrect, the result will differ.
    if not torch.allclose(output, expected, rtol=1e-5, atol=1e-5):
        pytest.skip("Kernel does not correctly handle remainder elements when size is not divisible by 4.")
    else:
        pytest.fail("Kernel unexpectedly produced the correct result for non-divisible input size, " +
                    "which should trigger a bug in more general cases.")

# Issue 4: Test that the kernel launch does not report any cuda errors.
def test_kernel_error_checking():
    # Force an error by providing an input tensor on the CPU to a CUDA kernel.
    input_tensor = torch.randn(512, dtype=torch.float32, device="cpu")
    module = build_kernel()
    with pytest.raises(RuntimeError):
        # This should raise an error because the CUDA kernel expects a CUDA tensor.
        output = module.forward(input_tensor)
        torch.cuda.synchronize()
