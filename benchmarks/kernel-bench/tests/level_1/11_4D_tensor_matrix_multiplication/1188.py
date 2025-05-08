
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="einsum_kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return cuda_module

# --- Issue 1: Data type is not float32 ---
def test_non_float_tensor():
    # Create inputs with double precision
    b, i, j, l, k = 2, 4, 4, 3, 5
    A = torch.randn(b, i, j, l, device="cuda", dtype=torch.float64)
    B = torch.randn(l, k, device="cuda", dtype=torch.float64)
    # The kernel expects float32. When given double input, it reinterprets each 8-byte
    # value as two 4-byte floats. Thus the kernel output will be wrong.
    module = build_kernel()
    C = module.forward(A, B)
    # Compute reference result with torch.einsum (which correctly handles double)
    C_ref = torch.einsum("bijl,lk->bijk", A, B)
    # The outputs should differ significantly.
    difference = (C - C_ref).abs().max().item()
    assert difference > 1e-2, (
        f"Expected significant difference due to data type mismatch, but max difference is {difference}"
    )

# --- Issue 2: Non-contiguous input ---
def test_non_contiguous_input():
    # Create contiguous inputs first
    b, i, j, l, k = 2, 4, 4, 3, 5
    A = torch.randn(b, i, j, l, device="cuda", dtype=torch.float32)
    B = torch.randn(l, k, device="cuda", dtype=torch.float32)
    # Make A non-contiguous by transposing two dimensions.
    A_noncontig = A.transpose(1, 2)
    # Even though A_noncontig and A have the same shape logically,
    # the underlying memory layout is not compatible with the kernel's indexing.
    module = build_kernel()
    C = module.forward(A_noncontig, B)
    C_ref = torch.einsum("bijl,lk->bijk", A_noncontig.contiguous(), B)
    # Expect significant differences because the kernel assumes contiguous layout.
    difference = (C - C_ref).abs().max().item()
    assert difference > 1e-2, (
        f"Kernel did not trigger an error with non-contiguous input (max diff: {difference})."
    )

# --- Issue 3: Missing error check for kernel launch ---
def test_kernel_launch_error():
    # Intentionally provide mismatched dimensions to trigger a TORCH_CHECK failure.
    # Here, we make the last dimension of A not match the first dimension of B.
    b, i, j, l, k = 2, 4, 4, 3, 5
    A = torch.randn(b, i, j, l, device="cuda", dtype=torch.float32)
    # Provide a B whose first dimension does not equal l.
    B = torch.randn(l + 1, k, device="cuda", dtype=torch.float32)
    module = build_kernel()
    with pytest.raises(RuntimeError, match="Dimension mismatch in l"):
        module.forward(A, B)
