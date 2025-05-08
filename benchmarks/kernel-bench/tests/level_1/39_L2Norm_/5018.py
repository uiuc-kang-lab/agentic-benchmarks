
import torch
import pytest
from torch.utils.cpp_extension import load
import os

def build_kernel():
    # We force a rebuild for testing
    path = os.path.dirname(os.path.realpath(__file__))
    cuda_module = load(
        name="l2norm_module",
        sources=[os.path.join(path, "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Trigger misalignment / non-contiguous memory issue.
# We create a tensor that is non-contiguous along the normalization dimension.
def test_non_contiguous_input():
    torch.manual_seed(0)
    batch_size = 16
    dim = 16384
    # Create a larger tensor then slice with a step to force non-contiguity
    base = torch.randn(batch_size, dim * 2, device="cuda", dtype=torch.float32)
    # Slicing every second element yields the correct number of columns but non-contiguous memory
    x = base[:, ::2]
    # Sanity-check: x is non-contiguous
    assert not x.is_contiguous(), "Input tensor is unexpectedly contiguous."

    my_kernel = build_kernel()
    # Run the CUDA kernel via the PyTorch binding
    y = my_kernel.forward(x)
    # Reference computed with PyTorch normalization along dim=1
    norm = x.norm(p=2, dim=1, keepdim=True)
    y_ref = x / (norm + 1e-12)
    
    # We expect a discrepancy because the kernel’s pointer arithmetic assumes contiguous memory.
    # If the kernel were general, the outputs would match.
    diff = (y - y_ref).abs().max().item()
    assert diff > 1e-4, f"Non-contiguous input did not trigger any noticeable error (max diff {diff})."

# Test 2: Trigger indexing issues for multi-dimensional inputs.
# The kernel is written assuming a 2D tensor normalized along dim=1.
# We create a 3D tensor that the kernel is not designed for.
def test_multi_dimensional_input():
    torch.manual_seed(0)
    batch_size = 8
    dim = 1024
    extra = 5  # extra dimension to simulate a more complex tensor layout
    # Create a 3D tensor; our kernel will only look at dim 1 (the second dimension)
    x = torch.randn(batch_size, dim, extra, device="cuda", dtype=torch.float32)
    # The kernel uses input.size(1) as dimension C and input.stride(0) for base offset.
    # For a 3D tensor these do not yield the proper element indexing.
    my_kernel = build_kernel()
    y = my_kernel.forward(x)
    # Compute reference normalization on dim=1.
    norm = x.norm(p=2, dim=1, keepdim=True)
    y_ref = x / (norm + 1e-12)
    
    diff = (y - y_ref).abs().max().item()
    assert diff > 1e-4, f"Multi-dimensional input did not trigger any noticeable error (max diff {diff})."

# Test 3: Test with double precision to reveal potential issues with atomicAdd/warp reductions.
# Depending on the GPU architecture, atomicAdd on double might not be supported (or be slower).
def test_double_precision_input():
    torch.manual_seed(0)
    batch_size = 16
    dim = 2048
    x = torch.randn(batch_size, dim, device="cuda", dtype=torch.float64)
    my_kernel = build_kernel()
    
    # Run the kernel – if atomicAdd for double is problematic on the current device, the result will be off.
    y = my_kernel.forward(x)
    norm = x.norm(p=2, dim=1, keepdim=True)
    y_ref = x / (norm + 1e-12)
    
    diff = (y - y_ref).abs().max().item()
    # Expect that the output differs, triggering the issue.
    assert diff > 1e-5, f"Double precision input did not trigger any noticeable error (max diff {diff})."
    
if __name__ == "__main__":
    pytest.main([__file__])
