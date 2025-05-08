
import torch
import pytest
from torch.utils.cpp_extension import load
import numpy as np

def build_kernel():
    # Build the extension from kernel.cu in the current directory
    cuda_module = load(
        name="swish_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Expected swish activation implementation for reference
def swish_reference(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_non_divisible_by_4():
    # Create an input tensor with number of elements not divisible by 4.
    # For example, a tensor of shape (17,) has 17 elements (17 % 4 != 0)
    x = torch.randn(17, device="cuda", dtype=torch.float32)
    # Since the kernel only processes floor(17/4)*4 = 16 elements,
    # the final element in the output is uncomputed/undefined.
    module = build_kernel()
    y = module.forward(x)
    # Compute reference using PyTorch swish
    y_ref = swish_reference(x)
    # We expect the outputs to differ due to the kernel ignoring the last element.
    if torch.allclose(y, y_ref, atol=1e-5):
        pytest.fail("Kernel incorrectly processed non-divisible-by-4 inputs, but it should leave remainder uncomputed.")

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_non_float32_input():
    # Create an input tensor with type float64.
    # The kernel reinterprets the pointer as float4 and expects float32,
    # so using float64 will trigger incorrect computation.
    x = torch.randn(1024, device="cuda", dtype=torch.float64)
    module = build_kernel()
    try:
        y = module.forward(x)
    except RuntimeError:
        # If a runtime error is raised that is acceptable for wrong types.
        pytest.skip("Kernel does not support non-float32 input types, as expected.")
    # Compute reference using PyTorch swish (after casting to float32)
    y_ref = swish_reference(x.float())
    # The outputs are likely to be wrong
    if torch.allclose(y.float(), y_ref, atol=1e-5):
        pytest.fail("Kernel incorrectly processed non-float32 inputs, but it should produce wrong results due to misinterpretation of data.")

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_misaligned_input():
    # Create a tensor with extra element and then narrow it to simulate misalignment.
    # Although the tensor is contiguous, its data_ptr() might not be aligned on a 16-byte boundary.
    # We allocate an extra element to force a slice that shifts the underlying pointer.
    # Note: Alignment effects may depend on the allocator, but this is a potential trigger.
    base = torch.randn(1025, device="cuda", dtype=torch.float32)
    # narrow along the first dimension by skipping the first element.
    # The resulting tensor is contiguous but its underlying pointer is base.data_ptr() + sizeof(float)
    x = base.narrow(0, 1, 1024)
    # Ensure that the number of elements is divisible by 4 for full processing.
    if x.numel() % 4 != 0:
        x = x.narrow(0, 0, x.numel() - (x.numel() % 4))
    module = build_kernel()
    y = module.forward(x)
    y_ref = swish_reference(x)
    # The misaligned input is likely to result in an output that is not equal (or produces undefined behavior)
    if torch.allclose(y, y_ref, atol=1e-5):
        pytest.fail("Kernel incorrectly processed misaligned input tensor. Expected misaligned access to produce incorrect results.")
