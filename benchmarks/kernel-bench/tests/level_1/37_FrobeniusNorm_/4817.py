
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Utility function to compile the CUDA extension.
def build_kernel(extra_cuda_cflags=None):
    extra_cuda_cflags = extra_cuda_cflags or ["-O3", "--use_fast_math"]
    cuda_module = load(
        name="test_module",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=extra_cuda_cflags,
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# 1. Test for grid-stride loop issue:
# Create an extremely large tensor that forces the kernel launch grid to be 2D.
# With total blocks > 65535, if the kernel ignores blockIdx.y, then not all elements are normalized.
def test_grid_stride_issue():
    device = "cuda"
    # This tensor is huge to force a 2D grid configuration.
    # Note: The size is adjusted so that numel exceeds 65535 * BLOCK_SIZE.
    # BLOCK_SIZE is 512, so we want numel > 65535 * 512.
    num_elements = (65535 * 512) + 1000  
    # Make a 1D tensor and reshape to something arbitrary.
    x = torch.randn(num_elements, device=device, dtype=torch.float32)
    # We simulate a 2D input by reshaping
    x = x.view(1, -1)
    
    mod = build_kernel()
    output = mod.forward(x)
    # Compute reference normalization using PyTorch
    norm = torch.norm(x, p='fro')
    # If grid-stride loop were correct, these two should match.
    expected = x / norm
    # Because of missed elements the outputs will differ.
    assert not torch.allclose(output, expected, atol=1e-5), \
        "Test failed: Grid-stride loop issue not detected (unexpectedly matching outputs)."

# 2. Test for block reduction assumption on blockDim.x being a multiple of 32.
# We rebuild the kernel with BLOCK_SIZE set to a value not divisible by 32.
def test_blockdim_divisibility_issue():
    device = "cuda"
    # Use a small tensor so that grid configuration is simple.
    x = torch.randn(1024, device=device, dtype=torch.float32)
    # Provide a custom BLOCK_SIZE (e.g. 33 which is not divisible by 32) via extra_cuda_cflags.
    mod = build_kernel(extra_cuda_cflags=["-O3", "--use_fast_math", "-DBLOCK_SIZE=33"])
    output = mod.forward(x)
    # Compute reference normalization using PyTorch
    norm = torch.norm(x, p='fro')
    expected = x / norm
    # Because the reduction might be computed incorrectly, the output may differ.
    assert not torch.allclose(output, expected, atol=1e-5), \
         "Test failed: Block reduction issue not detected (unexpectedly matching outputs)."

# 3. Test for zero-division issue: input tensor with all zeros.
def test_zero_norm_issue():
    device = "cuda"
    # Create an input where all elements are zero.
    x = torch.zeros((16, 64, 256, 256), device=device, dtype=torch.float32)
    mod = build_kernel()
    output = mod.forward(x)
    # Division by zero should produce NaNs or Infs.
    assert torch.isnan(output).any() or torch.isinf(output).any(), \
         "Test failed: Division by zero did not produce NaN or Inf as expected."

# 4. Test for strict contiguous requirement.
def test_non_contiguous_input():
    device = "cuda"
    # Create a non contiguous tensor by transposing the last two dimensions.
    x = torch.randn((16, 64, 256, 256), device=device, dtype=torch.float32).transpose(2, 3)
    mod = build_kernel()
    with pytest.raises(RuntimeError, match="contiguous"):
        _ = mod.forward(x)

# 5. Test for strict float32 type requirement.
def test_input_dtype_issue():
    device = "cuda"
    # Create a tensor with dtype float64.
    x = torch.randn((16, 64, 256, 256), device=device, dtype=torch.float64)
    mod = build_kernel()
    with pytest.raises(RuntimeError, match="float32"):
         _ = mod.forward(x)
