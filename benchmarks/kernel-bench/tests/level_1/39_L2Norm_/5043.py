
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel(extra_cuda_cflags=None):
    cuda_module = load(
        name="l2_norm_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=extra_cuda_cflags if extra_cuda_cflags is not None else ["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Block shared memory overflow if blockDim.x > warpSize*32.
# In order to simulate a situation where the kernel is launched with too many threads,
# we add an extra compilation flag that overrides the hard‐coded thread count.
# (Assume that the kernel.cu is modified to use the macro OVERRIDE_THREADS if defined.)
def test_large_block_dimension():
    # This extra flag forces the kernel to use a large number of threads per block.
    extra_flags = ["-O3", "--use_fast_math", "-DOVERRIDE_THREADS=2048"]
    module = build_kernel(extra_cuda_cflags=extra_flags)
    # Use an input whose inner dimension (C) is large so that gridDim.y = ceil(C/threads) > 1.
    x = torch.randn(16, 16384, device="cuda", dtype=torch.float32)
    out = module.forward(x)
    # Calculate the L2 norm along dimension 1.
    norm = out.norm(p=2, dim=1, keepdim=True)
    # Because of the shared memory overflow the normalization is not correctly computed.
    # We expect the norm to deviate significantly from 1.
    assert not torch.allclose(norm, torch.ones_like(norm), atol=1e-3), \
        "Kernel with large block dimension appears to produce correct normalization; shared memory bound issue may not have been triggered."

# Issue 2: Lack of half precision support.
def test_half_precision():
    module = build_kernel()
    # Create a half precision input tensor.
    x = torch.randn(16, 16384, device="cuda", dtype=torch.half)
    # The kernel dispatch does not cover half so we expect a RuntimeError.
    with pytest.raises(RuntimeError):
        module.forward(x)

# Issue 3: Inflexible handling of input layout.
# The kernel assumes a 2D contiguous input (with normalization on dim 1).
# If given a non-contiguous tensor or one with additional dimensions the result may be incorrect.
def test_non_contiguous_and_higher_dimensional_input():
    module = build_kernel()
    # Create a 3D tensor and then make it non-contiguous by transposing two dimensions.
    x = torch.randn(4, 16, 16384, device="cuda", dtype=torch.float32)
    x = x.transpose(0, 1)  # new shape: (16, 4, 16384) and likely non-contiguous
    # The kernel uses input.size(1) as the normalization dimension.
    out = module.forward(x)
    # Compute the l2 norm along dimension 1 as the kernel would.
    norm = out.norm(p=2, dim=1, keepdim=True)
    # For correct L2 normalization we would expect the norm to be all ones.
    # Here we check that the result is (incorrectly) not normalized properly.
    assert not torch.allclose(norm, torch.ones_like(norm), atol=1e-3), \
        "Kernel returned nearly unit L2 norms even for non-contiguous or higher-dimensional inputs; expected mis-normalization due to fixed layout assumptions."

