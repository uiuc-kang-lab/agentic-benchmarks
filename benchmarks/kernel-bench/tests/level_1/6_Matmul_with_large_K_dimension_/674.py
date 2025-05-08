
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    # Rebuild the CUDA extension module on the fly
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

def _run_kernel(module, A, B):
    # Launch our custom kernel implemented in kernel.cu
    C = module.forward(A, B)
    # For safety, wait for device synchronization.
    torch.cuda.synchronize()
    return C

# Test case for Issue 1: Non-contiguous input tensors are not handled correctly.
def test_non_contiguous_inputs():
    # Create contiguous inputs first.
    M, K, N = 64, 128, 32
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    # Create a non-contiguous view for A by transposing it and then transposing back.
    # The resulting tensor has the same content as A but is non-contiguous.
    A_t = A.t().contiguous().t()  # Force non-contiguity: even though the numbers are the same,
                                  # the underlying strides are not the same as expected by the kernel.
    assert not A_t.is_contiguous(), "Test design error: Expected non-contiguous tensor for A."
    
    # Compute result using the custom kernel.
    module = build_kernel()
    C_kernel = _run_kernel(module, A_t, B)
    # Compute reference result using torch.matmul (which handles arbitrary strides).
    C_ref = torch.matmul(A_t, B)
    # Since our kernel is implemented under the assumption of contiguous row-major layout,
    # the result will be incorrect. We expect a significant difference.
    max_diff = (C_kernel - C_ref).abs().max().item()
    assert max_diff > 1e-3, f"Kernel did not fail on non-contiguous inputs as expected (max diff: {max_diff})."

# Test case for Issue 2: Half precision (float16) is not supported by the dispatch macro.
def test_half_precision_unsupported():
    M, K, N = 64, 128, 32
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    module = build_kernel()
    # The kernel dispatch only covers float32 and float64.
    with pytest.raises(RuntimeError, match="AT_DISPATCH_FLOATING_TYPES"):
        _ = _run_kernel(module, A, B)
