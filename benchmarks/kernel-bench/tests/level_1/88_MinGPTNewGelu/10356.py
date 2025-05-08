
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# Helper to compile and load the CUDA extension from 'kernel.cu'
def build_kernel():
    cuda_module = load(
        name="gelu_kernel_ext",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Passing a tensor of an unsupported type (e.g. float64) should trigger an issue.
def test_input_tensor_type():
    my_module = build_kernel()
    # Create a double precision tensor on CUDA.
    x = torch.randn(1024, device="cuda", dtype=torch.float64)
    # When passing a double tensor, the kernel computes with reinterpret_cast to float,
    # so the results will not match the expected GELU.
    y_kernel = my_module.forward(x)
    # Compute reference using PyTorch's GELU (convert input to float for correctness)
    # Note: we expect them to differ significantly.
    x_float = x.float()
    y_ref = F.gelu(x_float)
    # The kernel output, if computed, is in float; so compare with reference with relaxed tolerance.
    # The difference should be large because using double's bit pattern as float is an error.
    assert not torch.allclose(y_kernel, y_ref, atol=1e-3), "Kernel should not produce correct output for double tensor."

# Test 2: Passing a tensor that is contiguous but not 16-byte aligned to trigger misaligned accesses.
def test_misaligned_input():
    my_module = build_kernel()
    # Create a larger tensor and then take a narrow slice, which will be contiguous
    # but its data pointer is likely not 16-byte aligned.
    # Note: While many allocations are aligned, using a narrow view may force an offset.
    base = torch.randn(1025, device="cuda", dtype=torch.float32)
    # Take a slice starting from index 1. The resulting tensor is contiguous but may not be 16-byte aligned.
    x = base.narrow(0, 1, 1024)
    # Compute the kernel output.
    y_kernel = my_module.forward(x)
    # Compute reference using PyTorch's GELU implementation.
    y_ref = F.gelu(x)
    # Because the kernel uses vectorized loads/stores that assume 16-byte alignment,
    # misaligned input should produce results that deviate from the reference.
    if torch.allclose(y_kernel, y_ref, atol=1e-5):
        pytest.fail("Kernel output matches reference for a misaligned tensor, but it should trigger misaligned access issues.")
