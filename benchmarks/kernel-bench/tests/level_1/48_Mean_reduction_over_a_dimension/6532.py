
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the CUDA extension from the kernel.cu file.
    cuda_module = load(
        name="mean_reduce_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_non_contiguous_input():
    """
    Test the kernel with a non-contiguous input tensor.
    The kernel assumes contiguous memory; a non-contiguous tensor should cause
    an incorrect result.
    """
    module = build_kernel()
    # Create a contiguous tensor then make it non-contiguous by transposing dimensions.
    x = torch.randn(4, 8, 16, device='cuda', dtype=torch.float32)
    x_noncontig = x.transpose(0, 1)  # now non-contiguous
    dim = 1
    # The correct result computed by torch.mean is based on the non-contiguous view
    ref = torch.mean(x_noncontig, dim=dim)
    # Use our kernel; note that it does not handle non-contiguous memory correctly.
    out = module.forward(x_noncontig, dim)
    torch.cuda.synchronize()
    # The output will likely differ from the reference result.
    assert not torch.allclose(out, ref, atol=1e-5), "Kernel unexpectedly computed correct result with non-contiguous input."

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_invalid_dimension():
    """
    Test the kernel with an invalid reduction dimension.
    If the reduction dimension is out-of-bound, the kernel may behave unexpectedly.
    An exception or a wrong result is expected.
    """
    module = build_kernel()
    x = torch.randn(4, 8, 16, device='cuda', dtype=torch.float32)
    # Provide an invalid dimension.
    invalid_dim = 10  # x.dim() is 3, so 10 is out-of-bound.
    with pytest.raises(Exception):
        # Expecting either an exception from the kernel launch or from the host wrapper.
        _ = module.forward(x, invalid_dim)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_unsupported_dtype():
    """
    Test the kernel with an unsupported tensor dtype (e.g. int32).
    The kernel only dispatches for floating point types, so using int tensors 
    should cause an error.
    """
    module = build_kernel()
    x = torch.randint(0, 100, (4, 8, 16), device='cuda', dtype=torch.int32)
    dim = 1
    with pytest.raises(RuntimeError):
        _ = module.forward(x, dim)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_inner_tile_edge_case():
    """
    Test the kernel with an inner dimension that is not evenly divisible
    by TILE_INNER (which is hard-coded as 32). This will exercise the boundary
    check in the kernel and verify that the reduction result is computed correctly.
    """
    module = build_kernel()
    # Create a tensor where the 'inner' dimension is not a multiple of 32.
    # Here we consider an input shape of [outer, dim, inner] where inner=45.
    # For example, shape = [2, 10, 45] and we reduce along dim=1.
    x = torch.randn(2, 10, 45, device='cuda', dtype=torch.float32)
    dim = 1
    ref = torch.mean(x, dim=dim)
    out = module.forward(x, dim)
    torch.cuda.synchronize()
    assert torch.allclose(out, ref, atol=1e-5), f"Kernel output does not match reference output! Max error: {(out - ref).abs().max()}"
