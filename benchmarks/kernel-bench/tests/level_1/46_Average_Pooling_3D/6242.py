
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build the CUDA kernel extension; assumes kernel.cu is in the same directory.
def build_kernel():
    module = load(
        name="kernel_module",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return module

# Test 1: Trigger issue with unsupported data type (non float32).
def test_non_float32_input():
    kernel_module = build_kernel()
    # Create a double (float64) tensor on CUDA.
    # We expect that the kernel, which is written only for float type,
    # will fail or produce incorrect results.
    x = torch.randn(2, 2, 8, 8, 8, dtype=torch.double, device="cuda")
    kernel_size = 3
    stride = 2
    padding = 1
    # Due to type mismatch, the kernel should raise an error.
    with pytest.raises(RuntimeError):
        # This call should trigger a type-related error because the extension expects float.
        out = kernel_module.forward(x, kernel_size, stride, padding)
    torch.cuda.synchronize()

# Test 2: Trigger issue with non-contiguous input.
def test_non_contiguous_input():
    kernel_module = build_kernel()
    # Create a float32 tensor but then create a non-contiguous view.
    x = torch.randn(2, 2, 8, 8, 8, device="cuda", dtype=torch.float32)
    # Permute to produce non-contiguous tensor; the module assumes standard [N, C, D, H, W]
    x_non_contig = x.permute(0, 2, 3, 4, 1)
    # Now, convert back to contiguous expected shape by mistake (simulating a processing error)
    # but intentionally do not call .contiguous() to trigger the error.
    # Note: in many real scenarios the tensor might be non-contiguous; our kernel indexing would be wrong.
    with pytest.raises(RuntimeError):
        # We expect that a mis-aligned tensor memory layout might produce an error
        # or, if it runs, incorrect results compared to nn.AvgPool3d.
        out = kernel_module.forward(x_non_contig, 3, 2, 1)
    torch.cuda.synchronize()
