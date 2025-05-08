
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="reverse_cumsum_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

def torch_reverse_cumsum(x, dim):
    # Reference implementation using torch operations
    return torch.cumsum(x.flip(dim), dim=dim).flip(dim)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_negative_dim():
    # Issue: negative dim values are not handled by the kernel.
    kernel = build_kernel()
    x = torch.randn(4, 5, device="cuda")
    # Using negative dim should be handled by PyTorch indexing,
    # but the kernel directly accesses strides[dim] so -1 indexing in C++ (std::vector::operator[])
    # is undefined behavior. We expect an exception.
    with pytest.raises(IndexError):
        # The kernel expects a positive index. Passing a negative one should trigger a bounds error.
        kernel.forward(x, -1)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_invalid_dim():
    # Issue: The kernel does not check if 'dim' is out-of-bounds.
    kernel = build_kernel()
    x = torch.randn(3, 3, device="cuda")
    invalid_dim = 2  # valid dims are 0 and 1 only.
    with pytest.raises(IndexError):
        kernel.forward(x, invalid_dim)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_non_contiguous_tensor():
    # Issue: The heuristic based on min_stride may fail for non-contiguous inputs.
    kernel = build_kernel()
    # Create a tensor that is non-contiguous.
    # For example, transpose a contiguous tensor
    x = torch.randn(4, 5, device="cuda")
    x_noncontig = x.t()  # now shape is (5,4) but non-contiguous
    # Choose a dim that will force the kernel to follow the first branch if strides[dim] equals min_stride.
    # The reference implementation (using torch operations) should work correctly.
    dim = 0
    y_kernel = kernel.forward(x_noncontig, dim)
    y_ref = torch_reverse_cumsum(x_noncontig, dim)
    # Compare results; if the heuristic fails, these may not be close.
    assert torch.allclose(y_kernel, y_ref, atol=1e-5), \
        f"Kernel reverse_cumsum on non-contiguous tensor produced incorrect result."

