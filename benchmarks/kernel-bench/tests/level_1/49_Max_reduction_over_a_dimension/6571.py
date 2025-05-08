
import pytest
import torch
from torch.utils.cpp_extension import load

# Utility function to build the CUDA module.
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1: Non-contiguous tensor.
# The kernel assumes a contiguous flattened tensor.
def test_non_contiguous():
    # Create a contiguous tensor and then a non-contiguous view (by transposing)
    x = torch.randn(16, 256, 256, device='cuda')
    x_t = x.transpose(1, 2)  # non-contiguous view
    # Use reduction on dimension 1 (now non contiguous)
    # The host code would use:
    # outer_size = product(x_t.size(i) for i in range(1))
    # inner_size = product(x_t.size(i) for i >= 2)
    # Because of the altered memory layout our kernel indexing will be off.
    kernel = build_kernel()
    out_kernel = kernel.forward(x_t, 1)
    out_ref = torch.max(x_t, dim=1)[0]
    # We expect the outputs to differ (or be wrong) because of index errors.
    assert not torch.allclose(out_kernel, out_ref, atol=1e-4), \
        "Kernel unexpectedly produced the same result with non-contiguous input."

# Issue 2: Reduction over a border dimension.
# For example, reducing over dim=0 in a rank-2 tensor will lead to an out-of-bound access.
def test_rank2_reduce_dim0():
    # Create a simple 2D tensor of shape (3, 4). Reducing over dim 0.
    x = torch.randn(3, 4, device='cuda')
    kernel = build_kernel()
    # Reduction over dim=0 in a rank-2 tensor.
    # The host code computes outer_size = 1 and inner_size = 4 and dim_size = 3,
    # so the kernel will use base index = 1*3*4+inner_idx, i.e. starting at element 12,
    # which is out of bounds.
    # We check that the result does not match torch.max.
    out_kernel = kernel.forward(x, 0)
    out_ref = torch.max(x, dim=0)[0]
    assert not torch.allclose(out_kernel, out_ref, atol=1e-4), \
        "Kernel unexpectedly produced the correct result when reducing over dim=0 in a rank-2 tensor."

# Issue 3: The kernel uses a sequential loop over the reduction dimension.
# For a case where the reduced dimension is large, performance may be poor or the sequential loop might accumulate rounding drift.
# Here, we test with an exaggerated dim_size.
def test_sequential_reduction_performance():
    # This test doesn’t have an easy pass/fail numerical criterion,
    # but for a very long reduction dimension, the sequential loop may make the result drift from the reference.
    # We choose a tensor where the extreme contrast in reduction length is visible.
    outer = 8
    dim_reduction = 4096  # large reduction dimension
    inner = 16
    x = torch.randn(outer, dim_reduction, inner, device='cuda')
    kernel = build_kernel()
    out_kernel = kernel.forward(x, 1)
    out_ref = torch.max(x, dim=1)[0]
    # Although both computations are mathematically equivalent, floating point accumulation in a long sequential loop
    # may lead to differences (or slow performance). We check for a slight difference.
    diff = (out_kernel - out_ref).abs().max().item()
    assert diff > 1e-6, \
        "Kernel sequential reduction did not trigger a rounding or performance issue as expected for large dim sizes."

# Issue 4: Unsupported tensor data types.
# The dispatch macro only supports floating types and half.
def test_type_not_supported():
    # Create an integer tensor, which is not supported by AT_DISPATCH_FLOATING_TYPES_AND_HALF.
    x_int = torch.randint(0, 10, (16, 256, 256), device='cuda', dtype=torch.int32)
    kernel = build_kernel()
    with pytest.raises(RuntimeError):
        # Expecting that running the kernel with an integer tensor will raise an error.
        _ = kernel.forward(x_int, 1)
