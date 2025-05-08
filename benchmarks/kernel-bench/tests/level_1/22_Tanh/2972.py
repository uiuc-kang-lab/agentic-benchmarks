
import torch
import pytest
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

# Test case 1: Non-contiguous tensor input
# This test uses a transpose to force a non-contiguous allocation.
# The kernel incorrectly assumes contiguous (and aligned) memory, so this test is designed to trigger potential issues.
def test_non_contiguous_tensor():
    kernel_module = build_kernel()
    # Create a contiguous tensor and then transpose to break the contiguous property.
    x = torch.randn(128, 256, device='cuda', dtype=torch.float32)
    x_non_contig = x.t()  # Non-contiguous view
    # The results of applying the CUDA kernel might be undefined if the input is non-contiguous.
    # We do not know the expected behavior, so we check that the output does not match the correct tanh.
    output = kernel_module.forward(x_non_contig)
    torch.cuda.synchronize()
    expected = torch.tanh(x_non_contig)
    # The test is successful if the output is not equal to the reference output,
    # indicating that the kernel's assumption about contiguous memory led to wrong results.
    assert not torch.allclose(output, expected, atol=1e-5), (
        "Kernel unexpectedly produced correct results on non-contiguous input. "
        "This test is designed to trigger the issue with non-contiguous memory."
    )

# Test case 2: Unsupported half precision (float16) inputs
# This test creates a half tensor and expects the kernel launch to fail or raise an error,
# because the AT_DISPATCH_FLOATING_TYPES macro only covers float and double.
def test_half_precision():
    kernel_module = build_kernel()
    x = torch.randn(1024, device='cuda', dtype=torch.float16)
    with pytest.raises(RuntimeError):
        # Expecting dispatch failure or error due to unhandled data type
        _ = kernel_module.forward(x)
