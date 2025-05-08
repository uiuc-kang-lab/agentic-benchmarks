
import pytest
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

# Build/load the CUDA extension
def build_kernel():
    cuda_module = load(
        name="custom_maxpool2d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Helper: reference implementation using PyTorch's native MaxPool2d
def run_native_maxpool(input_tensor, kernel_size, stride, padding, dilation):
    maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    return maxpool(input_tensor)

# Issue 1: Using std::numeric_limits in device code.
# Although AT_DISPATCH_FLOATING_TYPES limits the types to floating points,
# we test with integer input which is not supported.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_unsupported_dtype():
    kernel_size = 2
    stride = 2
    padding = 1
    dilation = 3
    # Create an integer tensor on CUDA. The kernel was dispatched only for floating types,
    # so we expect either an error or undefined behavior.
    x = torch.randint(0, 100, (16, 32, 128, 128), device="cuda", dtype=torch.int32)
    cuda_module = build_kernel()
    with pytest.raises(RuntimeError):
        # Expect failure because the AT_DISPATCH_FLOATING_TYPES branch
        # does not handle int32 and the kernel will not dispatch.
        cuda_module.forward(x, kernel_size, stride, padding, dilation)

# Issue 2: The kernel is not written for a non-square window or asymmetric parameters.
# This test intentionally passes a non-square kernel (simulated by different effective dilation offsets)
# by comparing the native and custom versions. In a general situation the kernel should be extended.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_asymmetric_configuration():
    # For the purpose of the test, we simulate an asymmetric situation by manually altering
    # one of the parameters after creation. Since our wrapper only accepts one kernel_size,
    # this test documents the limitation.
    kernel_size = 2
    stride = 2
    padding = 1
    dilation = 3
    # Create input tensor.
    x = torch.randn(16, 32, 128, 128, device="cuda", dtype=torch.float32)
    cuda_module = build_kernel()
    # Custom kernel execution.
    out_custom = cuda_module.forward(x, kernel_size, stride, padding, dilation)
    # Native implementation. Note: native MaxPool2d does not accept non-square window specification either,
    # so we use the same parameters. The intent here is to highlight that the current kernel
    # is not designed for general asymmetric pooling.
    out_native = run_native_maxpool(x, kernel_size, stride, padding, dilation)
    # We deliberately check for differences. They might be very subtle but the idea is that for more
    # general configurations the custom kernel would need extension.
    assert not torch.allclose(out_custom, out_native, atol=1e-5), (
        "Custom kernel unexpectedly matches native output; asymmetric case should be handled explicitly."
    )

# Issue 3: Non-contiguous input is not handled correctly.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_non_contiguous_input():
    kernel_size = 2
    stride = 2
    padding = 1
    dilation = 3
    x = torch.randn(16, 32, 128, 128, device="cuda", dtype=torch.float32)
    # Make x non-contiguous by transposing two dimensions.
    x_noncontig = x.transpose(2, 3)
    assert not x_noncontig.is_contiguous(), "Test setup error: tensor is unexpectedly contiguous."
    cuda_module = build_kernel()
    out_custom = cuda_module.forward(x_noncontig, kernel_size, stride, padding, dilation)
    out_native = run_native_maxpool(x_noncontig, kernel_size, stride, padding, dilation)
    # The expectation is that the custom kernel will yield different (incorrect) results.
    # We check that the outputs do not match.
    assert not torch.allclose(out_custom, out_native, atol=1e-5), (
        "Custom kernel output matches native output on non-contiguous input, but incorrect indexing is expected."
    )

# Issue 4: Lack of kernel launch error checking.
# We simulate an error (e.g., by providing an input size that is too small such that the computed pooling window is invalid).
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_invalid_pooling_parameters():
    kernel_size = 10
    stride = 1
    padding = 0
    dilation = 1
    # Create a very small input.
    x = torch.randn(2, 8, 5, 5, device="cuda", dtype=torch.float32)
    cuda_module = build_kernel()
    # In the native implementation, pooling on such a small tensor will still run but produce a result of size 1x1.
    # The custom kernel might try to access out-of-bound memory if error checking is missing.
    # We expect either a crash (caught as a RuntimeError upon synchronization) or an output that does not match.
    out_custom = cuda_module.forward(x, kernel_size, stride, padding, dilation)
    # Force synchronization to trigger any launch errors.
    try:
        torch.cuda.synchronize()
    except RuntimeError as e:
        pytest.skip(f"Kernel launch failed as expected with error: {e}")
    out_native = run_native_maxpool(x, kernel_size, stride, padding, dilation)
    # If no runtime error was raised, then check for correctness.
    assert not torch.allclose(out_custom, out_native, atol=1e-5), (
        "Custom kernel output unexpectedly matches the native output even with invalid pooling parameters."
    )
