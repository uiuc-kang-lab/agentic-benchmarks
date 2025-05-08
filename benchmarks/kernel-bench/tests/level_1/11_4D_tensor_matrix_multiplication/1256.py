
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

# Test 1: Non-float32 tensor types (Issue 1)
def test_non_float32_type():
    my_module = build_kernel()
    b, i, j, l, k = 4, 8, 8, 16, 32
    # Create double (float64) tensors, which are not supported by the kernel.
    A = torch.randn(b, i, j, l, dtype=torch.double, device='cuda')
    B = torch.randn(l, k, dtype=torch.double, device='cuda')
    with pytest.raises(RuntimeError):
        C = my_module.forward(A, B)
        torch.cuda.synchronize()

# Test 2: Non-contiguous inputs (Issue 2)
def test_non_contiguous_inputs():
    my_module = build_kernel()
    b, i, j, l, k = 2, 4, 4, 8, 16
    A_base = torch.randn(b, i, j, l, device='cuda', dtype=torch.float32)
    # Make A non-contiguous by transposing some dimensions.
    A = A_base.transpose(1, 3)
    # B is created as contiguous.
    B = torch.randn(l, k, device='cuda', dtype=torch.float32)
    # Since kernel expects contiguous memory (its indexing assumes a specific layout),
    # the result will be wrong. We check that the kernel's output differs from the expected result.
    C_kernel = my_module.forward(A, B)
    C_ref = torch.einsum("bijk,lk->bijl", A.contiguous().view(b, l, j, i).transpose(1,3), B)
    # Because the layout assumed by the kernel doesn't match the actual memory layout,
    # the results are expected to be different.
    with pytest.raises(AssertionError):
        # Perform an allclose check (which should fail)
        assert torch.allclose(C_kernel, C_ref, atol=1e-5), "Kernel output is unexpectedly correct for non-contiguous input."

# Test 3: Large K causing grid dimension issues (Issue 3)
def test_large_K_exceed_grid_dimension():
    my_module = build_kernel()
    # The maximum grid dimension in y is typically 65535.
    # We choose K such that (K + TILE_K - 1) / TILE_K > 65535.
    # For TILE_K = 32, choose K = 32 * 70000 (which exceeds 65535).
    b, i, j, l = 1, 1, 1, 16
    K = 32 * 70000  # large K
    A = torch.randn(b, i, j, l, device='cuda', dtype=torch.float32)
    B = torch.randn(l, K, device='cuda', dtype=torch.float32)
    # Launching the kernel with such a huge grid may silently fail or error out.
    with pytest.raises(Exception):
        C = my_module.forward(A, B)
        torch.cuda.synchronize()

# Test 4: Passing CPU tensors (should error in forward function) (Extra robustness check)
def test_cpu_input_error():
    my_module = build_kernel()
    b, i, j, l, k = 2, 4, 4, 8, 16
    A = torch.randn(b, i, j, l, device='cpu', dtype=torch.float32)
    B = torch.randn(l, k, device='cpu', dtype=torch.float32)
    with pytest.raises(RuntimeError):
        C = my_module.forward(A, B)
        torch.cuda.synchronize()
