
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper: load the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="sum_reduce_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test case 1: Non-contiguous input tensor
def test_non_contiguous_input():
    # Create a contiguous tensor of shape (4, 8, 16)
    x = torch.randn(4, 8, 16, device="cuda", dtype=torch.float32)
    # Make it non-contiguous by transposing two dimensions (so that inner layout is altered)
    x_non_contig = x.transpose(1, 2)
    # We'll reduce along dimension = 1 (which is not contiguous in x_non_contig)
    reduce_dim = 1
    kernel = build_kernel()
    output_kernel = kernel.forward(x_non_contig, reduce_dim)
    # Expected output using PyTorch sum (which works for non-contiguous tensors)
    output_ref = torch.sum(x_non_contig, dim=reduce_dim, keepdim=True)
    # The kernel uses linear indexing assuming contiguous layout so the result will be incorrect.
    # We assert that the kernel output does NOT match the correct result.
    assert not torch.allclose(output_kernel, output_ref, atol=1e-5), (
        "Test failed to trigger non-contiguous input issue. Kernel output unexpectedly matches torch.sum."
    )

# Test case 2: Half-precision (float16) input tensor
def test_half_precision_not_supported():
    # Create a tensor of dtype float16. The kernel dispatch macro does not cover float16.
    x = torch.randn(16, 32, 32, device="cuda", dtype=torch.float16)
    reduce_dim = 1
    kernel = build_kernel()
    with pytest.raises(RuntimeError):
        # The kernel should either throw an error or produce incorrect results.
        # We choose to expect an error on unsupported type.
        _ = kernel.forward(x, reduce_dim)

# Test case 3: Empty reduction dimension (reduce_size == 0)
def test_empty_reduction_dim():
    # Create a tensor with the reduction dimension having size 0.
    # For example, shape (2, 0, 5) and we reduce on dim=1.
    x = torch.empty(2, 0, 5, device="cuda", dtype=torch.float32)
    reduce_dim = 1
    kernel = build_kernel()
    # The torch.sum result is defined for empty dimensions (returns 0 sums),
    # but our kernel has no explicit handling for reduce_size == 0.
    output_kernel = kernel.forward(x, reduce_dim)
    output_ref = torch.sum(x, dim=reduce_dim, keepdim=True)
    # We expect a discrepancy or possibly undefined behavior.
    assert not torch.allclose(output_kernel, output_ref, atol=1e-5), (
        "Kernel did not trigger an issue with empty reduction dimension when it should."
    )

# Test case 4: Input with non-canonical memory layout (general layout assumption)
def test_non_canonical_layout():
    # Create a tensor with shape (3, 4, 5) and then perform a view that makes it non-canonical.
    x = torch.randn(3, 4, 5, device="cuda", dtype=torch.float32)
    # For instance, permuting dimensions so that the reduction dimension is not in the middle
    # even though the kernel’s computation assumes a specific layout.
    x_permuted = x.permute(2, 0, 1)  # New shape (5, 3, 4). We'll reduce along dim=1.
    reduce_dim = 1
    kernel = build_kernel()
    output_kernel = kernel.forward(x_permuted, reduce_dim)
    output_ref = torch.sum(x_permuted, dim=reduce_dim, keepdim=True)
    assert not torch.allclose(output_kernel, output_ref, atol=1e-5), (
        "Kernel did not trigger an issue with non-canonical memory layout affecting index calculation."
    )
