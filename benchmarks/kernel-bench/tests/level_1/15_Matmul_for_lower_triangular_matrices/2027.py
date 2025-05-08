
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper function to compile and load the CUDA kernel.
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1: Floating-point inversion for block-to-tile mapping might be inaccurate for very large matrices.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_large_matrix_tile_index_inversion():
    # Construct a matrix size large enough to require many tiles.
    # (While triggering exact rounding failure is tricky, using a large size is likely to expose inaccuracies.)
    N = 8192  # This yields a high block count that relies on floating-point inversion.
    A = torch.randn(N, N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, N, device="cuda", dtype=torch.float32)
    A = torch.tril(A)
    B = torch.tril(B)

    mod = build_kernel()

    # Run the custom kernel multiplication.
    C = mod.forward(A, B)
    torch.cuda.synchronize()

    # Compute reference result using torch.matmul and torch.tril.
    C_ref = torch.tril(torch.matmul(A, B))

    # While floating point rounding in mapping may not cause a crash,
    # it could lead to small numerical differences.
    assert torch.allclose(C, C_ref, atol=1e-4), \
        f"Large matrix multiplication mismatch: max diff {(C - C_ref).abs().max().item()}"

# Issue 2: The kernel only supports float32; if a double tensor is provided the kernel misinterprets memory.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_incorrect_dtype():
    N = 128
    # Create double tensors.
    A = torch.randn(N, N, device="cuda", dtype=torch.float64)
    B = torch.randn(N, N, device="cuda", dtype=torch.float64)
    A = torch.tril(A)
    B = torch.tril(B)

    mod = build_kernel()

    with pytest.raises(RuntimeError):
        # The kernel will use A.data_ptr<float>() on a double tensor,
        # leading to memory corruption or an assertion failure from the TORCH_CHECKs.
        _ = mod.forward(A, B)

# Issue 3: The kernel assumes square, lower-triangular matrices.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_non_square_input():
    # Provide non-square input (e.g. more columns than rows)
    A = torch.randn(128, 130, device="cuda", dtype=torch.float32)
    B = torch.randn(128, 130, device="cuda", dtype=torch.float32)
    
    mod = build_kernel()

    with pytest.raises(RuntimeError) as excinfo:
        _ = mod.forward(A, B)
    assert "square" in str(excinfo.value), "Expected a square matrix check failure."

# Issue 4: The kernel assumes contiguous input, but non-contiguous inputs may give wrong results.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_non_contiguous_input():
    N = 256
    A = torch.randn(N, N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, N, device="cuda", dtype=torch.float32)
    A = torch.tril(A)
    B = torch.tril(B)
    
    # Make non-contiguous inputs by transposing (which for square matrices gives a non-contiguous view)
    A_nc = A.t()
    B_nc = B.t()
    
    mod = build_kernel()
    
    # Note: The kernel does not check contiguity, so it may produce wrong results.
    # Here we compare against a contiguous reference.
    C_nc = mod.forward(A_nc, B_nc)
    torch.cuda.synchronize()
    # For a lower-triangular matrix, transposition does not preserve the lower-triangular property.
    # Thus, we cannot use torch.tril(matmul) directly.
    # We force-contiguity and recompute reference.
    C_ref = torch.tril(torch.matmul(A_nc.contiguous(), B_nc.contiguous()))
    
    # The test expects a mismatch due to non-contiguous memory being used.
    # If the kernel were safely handling non-contiguous tensors, C_nc would equal C_ref.
    # So here we assert that they differ.
    if torch.allclose(C_nc, C_ref, atol=1e-5):
        pytest.fail("Kernel unexpectedly handled non-contiguous inputs correctly; expected misinterpretation.")

