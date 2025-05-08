
import pytest
import torch
import torch.nn.functional as F
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

# Test case 1: Trigger issue with non-float32 input
def test_non_float32_input():
    # Create a double tensor (float64) on the CUDA device.
    # The kernel expects float32, so this is an error in general.
    x = torch.randn(1024, device='cuda', dtype=torch.float64)
    module = build_kernel()
    # Call the kernel – even though the CHECK_INPUT macro passes (only checking cuda and contiguity),
    # the kernel uses reinterpret_cast as if x were float.
    # Because F.elu supports float64 while our kernel does not,
    # the result will differ from PyTorch's F.elu.
    result = module.forward(x)
    expected = F.elu(x)
    # We expect the results to be different because of the wrong dtype.
    with pytest.raises(AssertionError):
        torch.testing.assert_allclose(result, expected, atol=1e-4)

# Test case 2: Trigger issue with misaligned input
def test_misaligned_input():
    # Create a tensor that is originally properly allocated as float32.
    # Then create a sliced view that is contiguous but whose storage pointer
    # is offset relative to the original allocation. This offset may break
    # the float4 alignment assumption.
    base = torch.randn(1025, device='cuda', dtype=torch.float32)
    # Slicing off the first element produces a tensor that is contiguous
    # but whose data pointer is likely not 16-byte aligned.
    x = base.narrow(0, 1, 1024)
    module = build_kernel()
    result = module.forward(x)
    expected = F.elu(x)
    # Because misaligned loads may produce incorrect results, we check that
    # the output does not match the expected output. We expect an assertion error.
    with pytest.raises(AssertionError):
        torch.testing.assert_allclose(result, expected, atol=1e-4)
