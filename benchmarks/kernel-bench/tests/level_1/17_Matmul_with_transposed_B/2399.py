
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Helper function to build and load the CUDA extension module from kernel.cu.
def build_kernel():
    # Assume kernel.cu is in the same directory as this file.
    module = load(
        name="custom_matmul",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return module

# Issue 1 Test: Passing an input tensor with a type other than float32.
def test_non_float32_input():
    my_module = build_kernel()
    # Create double-precision tensors.
    M, K, N = 64, 128, 32
    A = torch.randn(M, K, dtype=torch.float64, device="cuda")
    B = torch.randn(N, K, dtype=torch.float64, device="cuda")
    # The kernel calls data_ptr<float>() so it will misinterpret the memory.
    # The result will likely be numerically incorrect.
    C = my_module.forward(A, B)
    # Compute the correct result using torch.matmul
    C_ref = torch.matmul(A, B.T)
    # We expect the result to deviate considerably.
    assert not torch.allclose(C, C_ref, atol=1e-3), "Kernel should fail when using non-float32 inputs."

# Issue 2 Test: Passing inputs where B is not provided in the expected layout.
def test_incorrect_B_layout():
    my_module = build_kernel()
    M, K, N = 64, 128, 32
    A = torch.randn(M, K, dtype=torch.float32, device="cuda")
    # Incorrectly supply B as (K, N) (i.e. already transposed from expectation)
    B_wrong = torch.randn(K, N, dtype=torch.float32, device="cuda")
    # The forward function checks that A.size(1)==B.size(1) so we need to adjust.
    # Instead, simulate a user mistake by manually transposing after creating a proper tensor.
    B = torch.randn(N, K, dtype=torch.float32, device="cuda")
    B_incorrect = B.T  # now B_incorrect shape is (K, N)
    with pytest.raises(RuntimeError):
        # This should trigger an error or produce a wrong result because dimensions don't match the assumed layout.
        C = my_module.forward(A, B_incorrect)
        torch.cuda.synchronize()

# Issue 3 Test: Simulating a situation where the hardware warp size is assumed to be 32.
# While we cannot change the hardware warp size in a test, we can simulate the consequence by checking
# that the reduction is not adaptable. For example, we can run with a K value that is not a multiple of 32.
def test_non_multiple_of_warp_size():
    my_module = build_kernel()
    M, K, N = 64, 130, 32  # K is 130, not a multiple of 32.
    A = torch.randn(M, K, dtype=torch.float32, device="cuda")
    B = torch.randn(N, K, dtype=torch.float32, device="cuda")
    C = my_module.forward(A, B)
    C_ref = torch.matmul(A, B.T)
    # Since the kernel reduction is hard-coded for 32 lanes, numerical error might occur when K is non-multiple.
    # We check that the maximum difference is beyond tolerance.
    max_diff = (C - C_ref).abs().max().item()
    assert max_diff > 1e-3, f"Kernel reduction likely assumed warp size 32; but difference {max_diff} is too small."

# Issue 4 Test: Using matrices whose dimensions force incomplete blocks.
def test_incomplete_blocks():
    my_module = build_kernel()
    # Choose dimensions such that M and N do not align with block and grid mapping.
    M, K, N = 37, 71, 23  # arbitrary sizes that are not multiples of blockDim components.
    A = torch.randn(M, K, dtype=torch.float32, device="cuda")
    B = torch.randn(N, K, dtype=torch.float32, device="cuda")
    C = my_module.forward(A, B)
    C_ref = torch.matmul(A, B.T)
    # We expect numerical error due to the rigid mapping and potential unhandled edge conditions.
    max_diff = (C - C_ref).abs().max().item()
    assert max_diff > 1e-3, f"Kernel mapping to warps may not handle edges correctly; difference {max_diff} is too low."

if __name__ == "__main__":
    pytest.main([__file__])
