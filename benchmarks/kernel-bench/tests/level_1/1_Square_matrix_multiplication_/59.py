
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper function to build and return our CUDA module.
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1: Incorrect output due to wrong shared memory indexing (transposed access for matrix B).
# This test will create two square matrices and compare the kernel's output with torch.matmul.
def test_incorrect_output_due_to_transposed_indexing():
    # Choose a size that is not a multiple of TILE_SIZE to also stress the tiling boundaries.
    N = 65  # 65 is not a multiple of 16
    # Use float32 as required by the kernel
    A = torch.randn(N, N, device='cuda', dtype=torch.float32)
    B = torch.randn(N, N, device='cuda', dtype=torch.float32)
    mod = build_kernel()
    C_kernel = mod.forward(A, B)
    torch.cuda.synchronize()
    C_expected = torch.matmul(A, B)
    # The kernel is expected to produce wrong results because of the bug
    # so we assert that mismatches are indeed present.
    max_diff = (C_kernel - C_expected).abs().max().item()
    assert max_diff > 1e-3, (
        f"Expected kernel result to differ from torch.matmul due to incorrect indexing, "
        f"but max difference was {max_diff}"
    )

# Issue 2: The kernel only supports square matrices. This test will attempt to pass non-square inputs
# and expect an exception.
def test_non_square_matrix_error():
    # Create non-square matrices
    A = torch.randn(20, 30, device='cuda', dtype=torch.float32)
    B = torch.randn(30, 40, device='cuda', dtype=torch.float32)
    mod = build_kernel()
    # We expect the kernel to throw an error due to TORCH_CHECK enforcing square matrices
    with pytest.raises(RuntimeError) as exc_info:
        mod.forward(A, B)
    # Check that the error message contains a reference to the square matrix check.
    assert "square" in str(exc_info.value).lower(), (
        "Expected error message to complain about non-square input matrices."
    )
    
if __name__ == "__main__":
    pytest.main([__file__])
