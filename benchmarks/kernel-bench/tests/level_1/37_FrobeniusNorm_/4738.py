
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="custom_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Reduction kernel assumes a blockDim.x of 256.
# While the forward() function currently launches with 256 threads per block,
# this test forces a situation with a tensor size that does not nicely complement a 256-thread block.
# In a more general setup, if someone launches the kernel with a different block dimension,
# the reduction would be performed out-of-bounds.
# Since our module hard-codes threads=256, we simulate a condition where only very few elements are used.
def test_fixed_block_size_assumption():
    module = build_kernel()
    # Create a tensor with very few elements so that most threads in the fixed 256 threads block do no work.
    # Even though the kernel launch uses 256 threads, the reduction logic will access indices [tid + 128] even 
    # when there is little work.  We then check that the computed norm is as expected.
    x = torch.tensor([3.0, 4.0], device="cuda", dtype=torch.float32)  # norm should be 5.0
    out = module.forward(x)
    # Use a loose tolerance because atomic additions might lead to minor differences.
    assert torch.allclose(out, x / 5.0, atol=1e-4), "Block size assumption error in reduction kernel."

# Issue 2: Division by zero.
def test_division_by_zero():
    module = build_kernel()
    # Create a tensor that is entirely zero, resulting in norm=0 and a division-by-zero scenario.
    x = torch.zeros(1024, device="cuda", dtype=torch.float32)
    out = module.forward(x)
    # When dividing 0/0, the result is expected to be NaN.
    assert torch.isnan(out).all(), "Normalization of a zero tensor should result in NaNs due to division by zero."

# Issue 3: Unqualified use of min function might lead to compilation errors.
# While this error is a compile-time issue rather than a run-time error,
# we include a simple test to ensure that the module builds and can be invoked.
def test_module_compilation():
    module = build_kernel()
    # Try a simple random tensor normaliztion so that if there was a compilation problem (e.g. with min())
    # the module would not load.
    x = torch.randn(1024, device="cuda", dtype=torch.float32)
    out = module.forward(x)
    assert out is not None, "Module compilation failed, possibly due to an unqualified use of min()."
