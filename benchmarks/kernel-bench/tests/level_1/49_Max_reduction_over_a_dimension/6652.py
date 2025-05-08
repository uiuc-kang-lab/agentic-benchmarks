
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1: Non‑contiguous input
def test_non_contiguous_input():
    # Create a contiguous tensor and then make a non‐contiguous view via transposition.
    x = torch.randn(4, 8, 16, device="cuda")
    x_noncontig = x.transpose(1, 2)  # now non‐contiguous
    # Use torch.max to compute the reference reduction along dim=1.
    ref = torch.max(x_noncontig, dim=1)[0]
    module = build_kernel()
    # The kernel assumes contiguous memory; its result likely differs from ref.
    out = module.forward(x_noncontig, 1)
    torch.cuda.synchronize()
    # We expect a discrepancy because our kernel indexing is wrong for non‐contiguous data.
    assert not torch.allclose(out, ref), (
        "Kernel unexpectedly produced correct result for non-contiguous input; "
        "it should assume contiguous memory and thus fail for non-contiguous inputs."
    )

# Issue 2: Unsupported integer types
def test_integer_input():
    # Create an integer tensor.
    x_int = torch.randint(0, 100, (4, 8, 16), device="cuda", dtype=torch.int32)
    module = build_kernel()
    with pytest.raises(RuntimeError):
        # This should raise an error because the AT_DISPATCH macro does not cover int32.
        out = module.forward(x_int, 1)
        torch.cuda.synchronize()

# Issue 3: Invalid launch configuration via improper block_size
def test_invalid_block_size():
    x = torch.randn(4, 8, 16, device="cuda")
    module = build_kernel()
    with pytest.raises(RuntimeError):
        # Passing block_size=0 produces an invalid kernel launch configuration.
        out = module.forward(x, 1, block_size=0)
        torch.cuda.synchronize()

# Issue 4: Potential grid dimension limit violation
def test_grid_dimension_exceed():
    # Compute an inner_size so large that blocks_y = ceil(inner_size/block_size) will exceed
    # the CUDA grid dimension limit in y (typically 65535).
    # With block_size=256, choose inner_size = 256 * 65536, so that blocks_y becomes 65536.
    inner_size = 256 * 65536  # = 16777216
    # We construct a tensor with shape (1, 1, inner_size). The reduction is done over dim 1.
    x = torch.randn(1, 1, inner_size, device="cuda")
    module = build_kernel()
    with pytest.raises(RuntimeError):
        # The kernel launch should fail because grid.y exceeds the hardware maximum.
        out = module.forward(x, 1, block_size=256)
        torch.cuda.synchronize()
