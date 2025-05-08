
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Kernel only supports float32 inputs.
def test_dtype_failure():
    # Create double precision inputs to trigger wrong dtype error.
    M, N, K = 64, 16, 32
    # Construct tensors with float64; the kernel is written for float32.
    A = torch.randn(M, K, dtype=torch.float64, device="cuda")
    B = torch.randn(K, N, dtype=torch.float64, device="cuda")
    
    mod = build_kernel()
    with pytest.raises(Exception):
        # Expect an exception or incorrect behavior.
        mod.forward(A, B)
        
# Issue 2: Kernel assumes contiguous memory layouts.
def test_non_contiguous_input():
    # Create contiguous inputs first.
    M, N, K = 128, 16, 64
    A = torch.randn(M, K, dtype=torch.float32, device="cuda")
    B = torch.randn(K, N, dtype=torch.float32, device="cuda")
    # Force non-contiguity by transposing the matrices
    A_nc = A.t()  # This makes A_nc non-contiguous and changes its shape to (K, M)
    B_nc = B.t()  # Changes shape to (N, K)
    
    mod = build_kernel()
    # Depending on the branch taken in the CPU wrapper, these dimensions can be interpreted as transposed inputs.
    # We expect the non-contiguous layout to cause incorrect results.
    try:
        C = mod.forward(A_nc, B_nc)
        torch.cuda.synchronize()
        C_ref = torch.matmul(A_nc, B_nc)
        # The result may be wrong if the non-contiguity is not handled; so we check that they do not match.
        assert not torch.allclose(C, C_ref, atol=1e-5), (
            "Kernel unexpectedly produced correct output with non-contiguous inputs."
        )
    except Exception:
        # If an exception is thrown, that is acceptable too.
        pass

# Issue 3: No error checking on CUDA API calls (e.g. handling invalid dims).
def test_invalid_input_dimension():
    # Create 3D tensors to trigger the dim check in the CPU wrapper.
    A = torch.randn(4, 4, 4, dtype=torch.float32, device="cuda")
    B = torch.randn(4, 4, 4, dtype=torch.float32, device="cuda")
    
    mod = build_kernel()
    with pytest.raises(Exception):
        mod.forward(A, B)
