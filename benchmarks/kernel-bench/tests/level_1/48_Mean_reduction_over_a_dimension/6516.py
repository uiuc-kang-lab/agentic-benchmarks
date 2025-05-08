
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="mean_reduce_cuda_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_non_contiguous_input():
    """
    Test that triggers the issue when using a non-contiguous tensor.
    The kernel assumes contiguous layout, so using a non-contiguous tensor (obtained via transpose)
    should produce an incorrect mean computation when compared against torch.mean.
    """
    module = build_kernel()
    # Create a contiguous tensor and then make it non-contiguous by transposing
    x = torch.randn(8, 16, 32, device='cuda', dtype=torch.float32)
    # Transpose to get a non-contiguous tensor. Let's reduce along dimension 1 of the non-contiguous tensor.
    x_noncontig = x.transpose(1, 2)  # shape becomes (8, 32, 16), non-contiguous
    dim_to_reduce = 1
    # Compute result using our custom kernel and torch.mean
    out_kernel = module.forward(x_noncontig, dim_to_reduce)
    out_ref = torch.mean(x_noncontig, dim=dim_to_reduce)
    
    # The kernel is likely to produce a wrong result due to non-contiguous input.
    # We assert that the results are NOT close to flag the issue.
    assert not torch.allclose(out_kernel, out_ref, atol=1e-5), (
        "Custom CUDA kernel unexpectedly produced the correct result on a non-contiguous tensor."
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_half_precision_input():
    """
    Test that triggers the issue when using a float16 input.
    The kernel only dispatches on float32 and float64 types.
    Passing a float16 tensor should result in an error.
    """
    module = build_kernel()
    x_half = torch.randn(8, 16, 32, device='cuda', dtype=torch.float16)
    dim_to_reduce = 1
    with pytest.raises(RuntimeError):
        # Expecting a runtime error because float16 is not handled by AT_DISPATCH_FLOATING_TYPES.
        module.forward(x_half, dim_to_reduce)
