
import pytest
import torch
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

# Test 1: Trigger issue with wrong input tensor data type.
def test_wrong_dtype():
    my_module = build_kernel()
    # Create a tensor in double precision.
    x = torch.randn(16, 16384, device="cuda", dtype=torch.double)
    with pytest.raises(RuntimeError):
        # Expect the CHECK_INPUT or later reinterpret_cast to fail.
        out = my_module.forward(x)
        torch.cuda.synchronize()

# Test 2: Trigger issue with potential misalignment.
# We create a tensor with a non-zero storage offset that is still reported as contiguous.
def test_non_aligned_tensor():
    my_module = build_kernel()
    # Allocate a base tensor with extra elements.
    base = torch.randn(16 * 16384 + 1, device="cuda", dtype=torch.float32)
    # Create a sub-tensor starting from offset 1.
    # Even though the sub-tensor is contiguous (stride=1), its data pointer is misaligned.
    x = base.narrow(0, 1, 16 * 16384).view(16, 16384)
    # Although x.is_contiguous() is True, its underlying pointer is offset.
    # This should trigger misaligned vectorized loads in the kernel.
    # We do not expect correct results.
    out = my_module.forward(x)
    torch.cuda.synchronize()
    # Check that some elements differ from what F.elu would produce.
    out_ref = torch.nn.functional.elu(x, alpha=1.0)
    # If the kernel were correct, allclose would succeed.
    assert not torch.allclose(out, out_ref, atol=1e-5), \
        "Kernel unexpectedly produced correct results for a misaligned tensor."

# Test 3: Trigger issue due to lack of kernel launch error checking.
# We deliberately pass a tensor that is non-contiguous.
def test_non_contiguous_tensor():
    my_module = build_kernel()
    # Create a contiguous tensor first.
    x = torch.randn(16, 16384, device="cuda", dtype=torch.float32)
    # Make it non-contiguous by transposing.
    x_noncontig = x.t()
    # The CHECK_INPUT in the kernel should raise an error due to non-contiguity.
    with pytest.raises(RuntimeError):
        out = my_module.forward(x_noncontig)
        torch.cuda.synchronize()
