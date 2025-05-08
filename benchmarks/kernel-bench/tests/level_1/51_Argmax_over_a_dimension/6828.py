
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="efficient_argmax_kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.fixture(scope="module")
def cuda_module():
    return build_kernel()

# Issue 1: Only float32 inputs are supported.
def test_float_input_dtype(cuda_module):
    # Create a tensor with double type. This should trigger a TORCH_CHECK error.
    x_double = torch.randn(16, 10, 10, dtype=torch.double, device='cuda')
    with pytest.raises(RuntimeError, match="Only float32 is supported"):
        _ = cuda_module.forward(x_double, 1)

# Issue 2: Tie-breaking behavior for equal values.
def test_tie_breaking(cuda_module):
    # Create an input tensor where along the reduction dimension all values are the same.
    # Torch.argmax returns the index 0 (first occurrence) in this case.
    batch, dim, inner = 4, 10, 8
    x = torch.ones(batch, dim, inner, dtype=torch.float32, device='cuda')
    # The kernel is launched to reduce over dim=1.
    # Expected output shape: [batch, inner] and all entries should be 0.
    indices = cuda_module.forward(x, 1)
    expected = torch.zeros(batch, inner, dtype=torch.long, device='cuda')
    # If the reduction does not preserve the first occurrence due to the warp-level reduction,
    # the computed indices may not all be 0.
    assert torch.all(indices == expected), f"Expected all indices to be 0 (first occurrence), got {indices}"

# Issue 3: Hardcoded block size and shared memory assumptions.
def test_large_reduction_dimension(cuda_module):
    # Create a tensor with a large reduction dimension (dim=1) so that dimSize >> blockDim.x.
    # This tests whether the kernel can correctly handle multiple iterations in the reduction loop.
    batch, dim, inner = 2, 1024, 4  # 1024 is much larger than the hardcoded block size of 128.
    # We initialize the tensor such that the maximum occurs at a known index for each (batch, inner) pair.
    x = torch.randn(batch, dim, inner, dtype=torch.float32, device='cuda')
    # For each (batch, inner) pair, set index 537 to a high value.
    x[:, 537, :] = 1e6
    indices = cuda_module.forward(x, 1)
    # For each (batch, inner) pair, the maximum should be at index 537.
    expected = torch.full((batch, inner), 537, dtype=torch.long, device='cuda')
    assert torch.equal(indices, expected), f"Expected indices to be 537 for the max value, got {indices}"
