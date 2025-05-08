
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the CUDA kernel from the file kernel.cu.
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# 1. Test that passing a tensor with a non-float32 dtype produces an error.
def test_non_float_dtype():
    my_module = build_kernel()
    # Create double precision tensors
    A = torch.randn(64, 64, dtype=torch.double, device='cuda')
    B = torch.randn(64, 64, dtype=torch.double, device='cuda')
    with pytest.raises(Exception):
        # Should throw an error because the kernel expects float*
        C = my_module.forward(A, B)
        torch.cuda.synchronize()

# 2. Test that kernel launch errors are caught.
#    Here we purposely provide matrices with dimensions that are incompatible for multiplication.
def test_incompatible_dimensions():
    my_module = build_kernel()
    # A is 32x16 and B is 32x32, which are incompatible.
    A = torch.randn(32, 16, dtype=torch.float32, device='cuda')
    B = torch.randn(32, 32, dtype=torch.float32, device='cuda')
    with pytest.raises(Exception):
        C = my_module.forward(A, B)
        torch.cuda.synchronize()

# 3. Test that the “warp_optimized” kernel correctly handles transposed inputs.
#    Given the fragile handling of transposition, we test one of the transposed cases.
def test_transposition_logic():
    my_module = build_kernel()
    # Create a scenario where A should be interpreted as transposed.
    # Let A be a tall-skinny matrix stored in a transposed view.
    # For instance, if originally A is 16x64 and we want to treat it as 64x16.
    A_orig = torch.randn(16, 64, dtype=torch.float32, device='cuda')
    B = torch.randn(64, 32, dtype=torch.float32, device='cuda')
    # We simulate the transposed case by passing A.t() so that:
    # - A.t() has size (64, 16), and per our kernel logic this situation might force the transA flag.
    A = A_orig.t().contiguous()
    # Compute reference result using torch.matmul.
    C_ref = torch.matmul(A, B)
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    # Even a small numerical difference might highlight indexing issues.
    assert torch.allclose(C, C_ref, atol=1e-5), f"Transposition handling error: max diff = {(C - C_ref).abs().max()}"

# 4. Test to check naming inconsistency in the test harness (module naming error)
#    This test simulates the mistake by intentionally calling the wrong module variable.
def test_module_variable_name_error():
    my_module = build_kernel()
    # Create small compatible matrices.
    A = torch.randn(32, 32, dtype=torch.float32, device="cuda")
    B = torch.randn(32, 32, dtype=torch.float32, device="cuda")
    # Purposely use a wrong variable name to simulate the error found in the sample.
    with pytest.raises(NameError):
        # The variable "cuda_module" is not defined because we are using "my_module"
        C = cuda_module.forward(A, B)
        torch.cuda.synchronize()

if __name__ == "__main__":
    pytest.main([__file__])
