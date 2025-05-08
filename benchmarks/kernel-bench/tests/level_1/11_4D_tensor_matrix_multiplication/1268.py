
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Utility function to build the CUDA kernel extension.
def build_kernel(extra_cuda_cflags=None, extra_include_paths=None):
    extra_cuda_cflags = extra_cuda_cflags if extra_cuda_cflags is not None else ["-O3", "--use_fast_math"]
    module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=extra_cuda_cflags,
        extra_include_paths=extra_include_paths if extra_include_paths else [],
        with_cuda=True,
        verbose=False,
    )
    return module

# Test 1: Trigger issue with unsupported dtype. The kernel expects float32.
def test_wrong_dtype():
    my_module = build_kernel()
    b, i, j, l, k = 2, 8, 8, 16, 10
    # Create double tensors
    A = torch.randn(b, i, j, l, dtype=torch.float64, device="cuda")
    B = torch.randn(l, k, dtype=torch.float64, device="cuda")
    # The forward function does not check dtypes so it will reinterpret the raw bytes.
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    # Compute the reference using torch.einsum (which is correct)
    C_ref = torch.einsum("bijl,lk->bijk", A, B)
    # Expect that the results do not match because of misinterpreting dtypes.
    assert not torch.allclose(C, C_ref, atol=1e-5), (
        "Kernel should not work correctly with double tensors, but the output matches the reference!"
    )

# Test 2: Use non-contiguous input for A.
def test_non_contiguous_A():
    my_module = build_kernel()
    b, i, j, l, k = 2, 8, 8, 16, 10
    A = torch.randn(b, i, j, l, dtype=torch.float32, device="cuda")
    B = torch.randn(l, k, dtype=torch.float32, device="cuda")
    # Make A non-contiguous by transposing two dimensions, then flattening.
    A_non_contig = A.transpose(1, 3)  # Now shape is (b, l, j, i); non contiguous relative to original layout.
    # Pass the non-contiguous tensor to the kernel.
    C = my_module.forward(A_non_contig, B)
    torch.cuda.synchronize()
    # Compute reference using einsum on the non-contiguous tensor (which will force contiguity internally)
    C_ref = torch.einsum("blji,lk->bijk", A_non_contig, B)
    # The kernel assumes a contiguous layout with shape (b, i, j, l) so the results will be wrong.
    assert not torch.allclose(C, C_ref, atol=1e-5), (
        "Kernel did not fail as expected when A is non-contiguous."
    )

# Test 3: Trigger issue with wrong thread configuration.
# We simulate this by building the module with a different TILE_K macro definition.
# For example, we force TILE_K to a value (e.g., 64) that does not match the intended launch configuration.
def test_wrong_thread_configuration():
    # Build kernel with a redefinition of TILE_K.
    # By passing an extra flag, we override the macro to an unexpected value.
    my_module_wrong = build_kernel(extra_cuda_cflags=["-O3", "--use_fast_math", "-DTILE_K=64"])
    b, i, j, l, k = 2, 8, 8, 16, 10
    A = torch.randn(b, i, j, l, dtype=torch.float32, device="cuda")
    B = torch.randn(l, k, dtype=torch.float32, device="cuda")
    # When TILE_K is redefined, the kernel code uses the new macro value, but the host launch
    # in the forward() function still launches threads(dim=old TILE_K value as computed at compile time)
    # or it may mismatch the expected per-block thread count. This should lead to incorrect results.
    C = my_module_wrong.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.einsum("bijl,lk->bijk", A, B)
    assert not torch.allclose(C, C_ref, atol=1e-5), (
        "Kernel should produce incorrect results with a mismatched thread configuration (wrong TILE_K), but it did not!"
    )
