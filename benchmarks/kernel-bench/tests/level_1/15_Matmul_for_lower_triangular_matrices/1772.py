
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="triangular_mm",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.fixture(scope="module")
def kernel_module():
    return build_kernel()

# Issue 1: The kernel only supports float32 inputs.
def test_dtype_support(kernel_module):
    N = 64
    A = torch.randn(N, N, dtype=torch.float64, device='cuda')
    B = torch.randn(N, N, dtype=torch.float64, device='cuda')
    with pytest.raises(RuntimeError):
        # Expect a runtime error from TORCH_CHECK complaining about the tensor being non-CUDA float32.
        kernel_module.forward(A, B)

# Issue 2: Possible rounding error in warp-to-index mapping for large matrices.
def test_large_matrix_rounding(kernel_module):
    # Use a large matrix size where floating-point error in the inverse mapping might manifest.
    N = 4096
    A = torch.randn(N, N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, N, device="cuda", dtype=torch.float32)
    # Ensure the matrices are lower triangular
    A = torch.tril(A)
    B = torch.tril(B)
    
    C = kernel_module.forward(A, B)
    # Use the same computation as in the Python model: matmul then tril.
    C_ref = torch.tril(torch.matmul(A, B))
    # Allow a little tolerance to account for minor numerical variations.
    assert torch.allclose(C, C_ref, rtol=1e-3, atol=1e-3), \
        f"Kernel output does not match reference for a large matrix. Max diff: {(C - C_ref).abs().max().item()}"

# Issue 3: The kernel assumes contiguous inputs.
def test_non_contiguous_inputs(kernel_module):
    N = 128
    A = torch.randn(N, N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, N, device="cuda", dtype=torch.float32)
    # Create non-contiguous versions of A and B.
    # For example, by transposing twice with an intervening narrow slicing operation.
    A_nc = A[:, ::2].transpose(0, 1).contiguous().transpose(0, 1)
    B_nc = B[:, ::2].transpose(0, 1).contiguous().transpose(0, 1)
    # Make them lower triangular.
    A_nc = torch.tril(A_nc)
    B_nc = torch.tril(B_nc)
    
    C_nc = kernel_module.forward(A_nc, B_nc)
    # Compute reference using contiguous versions.
    C_ref = torch.tril(torch.matmul(A_nc.contiguous(), B_nc.contiguous()))
    assert torch.allclose(C_nc, C_ref, rtol=1e-5, atol=1e-5), \
        f"Kernel output does not match reference for non-contiguous inputs. Max diff: {(C_nc - C_ref).abs().max().item()}"
