
import pytest
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

def build_kernel():
    return load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

# A helper function to run our custom CUDA maxpool forward.
def cuda_maxpool(input, kernel_size, stride, padding, dilation):
    mod = build_kernel()
    return mod.forward(input, kernel_size, stride, padding, dilation)

# Test case for Issue 1:
# The kernel uses std::numeric_limits<scalar_t>::infinity() which may cause problems
# on some CUDA architectures. We simulate this by calling the kernel on float tensors
# and comparing its behavior against PyTorch’s built-in maxpool (both operating on negative numbers)
# In a correct implementation both should yield the same results. If the device constant is not set correctly,
# the kernel may propagate -infinity.
def test_numeric_limits_in_device_code():
    # Use parameters that may force some windows to have only negative numbers.
    batch_size, channels, height, width = 2, 3, 8, 8
    kernel_size, stride, padding, dilation = 2, 2, 1, 1
    # Create an input with values mostly negative
    input_tensor = -torch.abs(torch.randn(batch_size, channels, height, width, device='cuda', dtype=torch.float32))
    # Use PyTorch maxpool as reference.
    ref_maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    ref_out = ref_maxpool(input_tensor)
    # Use our CUDA kernel implementation.
    try:
        out = cuda_maxpool(input_tensor, kernel_size, stride, padding, dilation)
        torch.cuda.synchronize()
    except Exception as e:
        pytest.fail(f"Kernel failed with exception: {e}")

    # If the initialization using device infinity is wrong, output might contain -inf.
    assert not torch.isinf(out).any(), "Kernel output contains -infinity, which may indicate that std::numeric_limits<scalar_t>::infinity() was not correctly handled on device."

    # Also verify that our kernel produces nearly the same result as PyTorch’s maxpool.
    assert torch.allclose(out, ref_out, atol=1e-5), "Kernel output does not match PyTorch maxpool output."

# Test case for Issue 2:
# The kernel assumes a contiguous input layout.
# If passed a non-contiguous tensor, the linear index arithmetic will be invalid and the result will not match.
def test_non_contiguous_input():
    batch_size, channels, height, width = 2, 3, 10, 10
    kernel_size, stride, padding, dilation = 2, 2, 1, 1
    # Create a contiguous tensor then make it non-contiguous by transposing two dimensions.
    orig = torch.randn(batch_size, channels, height, width, device='cuda', dtype=torch.float32)
    non_contig = orig.transpose(2, 3)  # This makes it non-contiguous

    ref_maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    ref_out = ref_maxpool(non_contig)
    # Run our CUDA kernel on the non-contiguous tensor.
    try:
        out = cuda_maxpool(non_contig, kernel_size, stride, padding, dilation)
        torch.cuda.synchronize()
    except Exception as e:
        pytest.fail(f"Kernel failed with exception on non-contiguous input: {e}")

    # The outputs likely will differ because our kernel assumes contiguous layout.
    # Here we check that the outputs do not match, to flag the issue.
    if torch.allclose(out, ref_out, atol=1e-5):
        pytest.fail("Kernel incorrectly handled a non-contiguous input tensor.")

# Test case for Issue 3:
# The kernel uses AT_DISPATCH_FLOATING_TYPES which excludes half and integer types.
# Passing a half-precision tensor should result in a runtime error.
@pytest.mark.parametrize("dtype", [torch.float16, torch.int32])
def test_unsupported_dtype(dtype):
    batch_size, channels, height, width = 2, 3, 10, 10
    kernel_size, stride, padding, dilation = 2, 2, 1, 1
    input_tensor = torch.randn(batch_size, channels, height, width, device='cuda', dtype=dtype)
    with pytest.raises(RuntimeError) as excinfo:
        _ = cuda_maxpool(input_tensor, kernel_size, stride, padding, dilation)
    assert "AT_DISPATCH_FLOATING_TYPES" in str(excinfo.value), "Kernel did not complain about unsupported dtype as expected."
    
if __name__ == "__main__":
    pytest.main([__file__])
