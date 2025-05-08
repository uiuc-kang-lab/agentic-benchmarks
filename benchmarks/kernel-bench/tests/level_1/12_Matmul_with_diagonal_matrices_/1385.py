
import pytest
import torch
from torch.utils.cpp_extension import load

# Build/load the CUDA extension from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Test 1: Verify that the kernel misbehaves when input tensors are not float32.
# The kernel always casts the data pointer to float*, so passing float64 data 
# will produce incorrect results.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_dtype_incompatibility():
    my_module = build_kernel()
    N, M = 512, 512
    # Create float64 tensors instead of float32.
    A = torch.randn(N, device="cuda", dtype=torch.float64)
    B = torch.randn(N, M, device="cuda", dtype=torch.float64)
    # Run the kernel.
    C_kernel = my_module.forward(A, B)
    # Compute the expected result in float64.
    C_expected = torch.diag(A) @ B
    # The output from the kernel is computed incorrectly because the kernel 
    # used a float* pointer. We expect the maximum difference to be huge.
    max_diff = (C_kernel.double() - C_expected).abs().max().item()
    assert max_diff > 1e-2, f"Kernel unexpectedly produced nearly correct result with wrong dtype, max difference: {max_diff}"

# Test 2: Verify potential integer overflow issues for very large tensor sizes.
# This test attempts to create a tensor whose total number of elements is large enough
# (greater than what a 32-bit int can represent) so that the grid-stride loop indexing may overflow.
# Note, this test may be memory prohibitive on some systems; in that case, it is expected to skip.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_index_overflow_issue():
    my_module = build_kernel()
    # Choose dimensions such that total = N * M slightly exceeds the 32-bit integer range.
    # The maximum positive value representable in a signed 32-bit int is 2^31-1 = 2147483647.
    # We'll try to exceed that. Using square matrices: N = M = 46342 gives ~2.147e9 elements.
    N = 46342
    M = 46342
    try:
        A = torch.randn(N, device="cuda", dtype=torch.float32)
        B = torch.randn(N, M, device="cuda", dtype=torch.float32)
    except RuntimeError as e:
        pytest.skip("Skipping huge tensor test due to memory constraints.")
    
    C_kernel = my_module.forward(A, B)
    # Compute the expected result.
    C_expected = torch.diag(A) @ B
    # If integer overflow occurred in the kernel’s indexing, the kernel result will be widely off.
    max_diff = (C_kernel - C_expected).abs().max().item()
    # We expect a significant difference due to the index overflow issue.
    assert max_diff > 1e-3, f"Kernel output appears correct despite potential index overflow; max diff: {max_diff}"

# Test 3: Check behavior when input tensors are on CPU rather than CUDA.
# Since the extension does not check device locations, passing CPU tensors should lead to a runtime error.
def test_device_mismatch():
    my_module = build_kernel()
    N, M = 512, 512
    A = torch.randn(N, dtype=torch.float32)  # CPU tensor
    B = torch.randn(N, M, dtype=torch.float32)  # CPU tensor
    with pytest.raises(RuntimeError):
        _ = my_module.forward(A, B)
