
import pytest
import torch
from torch.utils.cpp_extension import load
import os

def build_kernel():
    # Build the CUDA extension from kernel.cu in the current directory.
    cuda_module = load(
        name="einsum_kernel_module",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1 test: unnecessary atomicAdd
# In the current design, every thread writes once. However, if one were to
# use the kernel in a situation where multiple threads write to the same output,
# the atomicAdd may hide race conditions. Here, we force a “collision” by launching
# with a very small J and K so that multiple blocks may update the same element.
def test_atomic_add_issue():
    # We choose dimensions deliberately small so that the grid may overlap in the batch/I dimension.
    # For instance, let I=1 so that blockIdx.z = batch * 1 gives only one row per batch,
    # but force more than one block to cover the same output element.
    b, i, j, l, k = 2, 1, 8, 16, 8  # very small spatial dims
    A = torch.randn(b, i, j, l, device="cuda", dtype=torch.float32)
    B = torch.randn(l, k, device="cuda", dtype=torch.float32)
    my_module = build_kernel()
    # Run kernel multiple times over the same output C (simulate accidental double accumulation)
    C1 = my_module.forward(A, B)
    C2 = my_module.forward(A, B)
    # If atomicAdd were not needed then a single launch would write the correct output.
    # Running twice artificially accumulates the value a second time.
    # So C2 should be (roughly) 2x C1. We check that.
    with torch.no_grad():
        diff = (C2 - 2 * C1).abs().max().item()
    # We expect nonzero difference (i.e. atomicAdd was used to accumulate rather than simply write).
    # (In the ideal design these two kernel calls would be independent writes, not additions.)
    assert diff > 1e-3, f"Unexpected result: the outputs do not show evidence of atomic accumulation. diff = {diff}"

# Issue 2 test: support for only float32
def test_hardcoded_float_issue():
    # Pass double tensors. The kernel does not check dtype so it will misinterpret the pointers.
    b, i, j, l, k = 2, 3, 4, 5, 6
    A = torch.randn(b, i, j, l, device="cuda", dtype=torch.float64)
    B = torch.randn(l, k, device="cuda", dtype=torch.float64)
    my_module = build_kernel()
    C = my_module.forward(A, B)
    # Compute a reference result using einsum after converting inputs to float32.
    C_ref = torch.einsum("bijl,lk->bijk", A.float(), B.float()).to(torch.float64)
    # Expected error: because the kernel is written for float32, the C result (interpreted as float32)
    # reinterpreted as float64 will not match the reference.
    max_diff = (C.to(torch.float64) - C_ref).abs().max().item()
    assert max_diff > 1e-3, f"Kernel accepted double inputs but produced correct result (max_diff={max_diff})"

# Issue 3 test: lack of generality (broadcasting, different einsum patterns)
def test_general_einsum_issue():
    # Try to trigger a situation where broadcasting is expected.
    # For example, let B be broadcastable over the batch dimension.
    b, i, j, l, k = 2, 3, 4, 5, 6
    A = torch.randn(b, i, j, l, device="cuda", dtype=torch.float32)
    # Instead of shape (l,k) we provide shape (1, l, k) expecting broadcasting along a new dimension.
    B = torch.randn(1, l, k, device="cuda", dtype=torch.float32)
    my_module = build_kernel()
    # The kernel checks only B.dim() == 2 so it should throw an error.
    with pytest.raises(RuntimeError):
        _ = my_module.forward(A, B)

# Issue 4 test: ineffective bank conflict avoidance in shared memory for B.
# This test does not easily “trigger” a failure in terms of correctness. However, if the shared memory padding
# were effective, one might expect improved performance on large inputs. Here we check that the kernel runs correctly 
# even if bank conflicts occur. (We assume that if bank conflicts were significant, the numerical results might suffer.)
def test_shared_memory_padding_issue():
    b, i, j, l, k = 2, 3, 32, 64, 32  # larger spatial dims
    A = torch.randn(b, i, j, l, device="cuda", dtype=torch.float32)
    B = torch.randn(l, k, device="cuda", dtype=torch.float32)
    my_module = build_kernel()
    C = my_module.forward(A, B)
    C_ref = torch.einsum("bijl,lk->bijk", A, B)
    max_diff = (C - C_ref).abs().max().item()
    # We expect correctness but performance might be degraded.
    assert max_diff < 1e-3, f"Kernel numerical result differs from reference (max_diff = {max_diff})"

# Issue 5 test: missing error checking / synchronization
# This test purposely launches the kernel with a mismatched inner dimension so that TORCH_CHECK fails.
def test_dimension_mismatch_error():
    b, i, j, l, k = 2, 3, 4, 5, 6
    A = torch.randn(b, i, j, l, device="cuda", dtype=torch.float32)
    # Create B with incorrect first dimension (l mismatch)
    B = torch.randn(l+1, k, device="cuda", dtype=torch.float32)
    my_module = build_kernel()
    with pytest.raises(RuntimeError):
        _ = my_module.forward(A, B)
