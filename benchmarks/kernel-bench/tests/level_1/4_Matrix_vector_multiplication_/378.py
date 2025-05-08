
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Utility: Rebuild the CUDA module.
def build_kernel():
    module = load(
        name="atomic_optimized_matvec",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

# Test 1: Trigger compilation/runtime errors due to __shared__ variable declared inside a loop.
# This test forces the kernel to iterate over multiple rows (i.e. grid-stride loop with M > gridDim.x)
# so that if the shared memory is mis-declared, it may trigger unexpected behavior.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_shared_declaration_issue():
    module = build_kernel()
    # Use M large enough to force the kernel loop over several rows.
    M = 300   # > 256 so that at least one block processes multiple rows.
    K = 1024
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, 1, device="cuda", dtype=torch.float32)
    C = module.forward(A, B)
    torch.cuda.synchronize()
    # We compute a reference result with torch.matmul
    C_ref = torch.matmul(A, B)
    # Relax the tolerance slightly since the atomic reductions could be imprecise as well,
    # but expect a significant deviation if the __shared__ error causes corruption.
    assert torch.allclose(C, C_ref, atol=1e-3), f"Output differs for multi-row grid iteration; possible __shared__ mis-declaration, max diff: {(C-C_ref).abs().max()}"

# Test 2: Trigger precision issues and potential lack of atomic support by using a floating point type 
# that is not natively supported (e.g. float16, notably if the hardware does not support atomicAdd on half).
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_dtype_atomic_support():
    module = build_kernel()
    M = 128
    K = 2048
    # Create inputs of type float16; some devices or compute caps do not support atomicAdd on half.
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, 1, device="cuda", dtype=torch.float16)
    try:
        C = module.forward(A, B)
        torch.cuda.synchronize()
    except RuntimeError as e:
        # Expecting a runtime error on devices that do not support atomicAdd for float16.
        pytest.skip("atomicAdd for float16 is not supported on this device; skipping test.")
    # For devices that do support atomicAdd on half, check approximate correctness.
    C_ref = torch.matmul(A.float(), B.float()).half()
    assert torch.allclose(C, C_ref, atol=1e-2), f"Kernel with float16 input produced results that diverge from reference."

# Test 3: Trigger issues due to strict assumption about input shape of B.
# This test provides B as a 2D column vector as expected by the high‐level PyTorch function but then
# internally the new kernel uses B.view({-1}), which might not work correctly in a general scenario.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_input_shape_assumption():
    module = build_kernel()
    M = 256
    K = 512
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    # Instead of a column vector, provide B as a 2D tensor with shape (K, 1) (which is expected by the model)
    B = torch.randn(K, 1, device="cuda", dtype=torch.float32)
    C = module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    assert torch.allclose(C, C_ref, atol=1e-5), f"Kernel output differs; possible misuse of input shape assumptions in kernel."

# Test 4: Check that the kernel handles cases when multiple blocks update the same output element.
# This test sets the grid to use fewer blocks than rows to force multi-block updates and rely on global atomicAdd.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_multi_block_updates():
    module = build_kernel()
    M = 500  # More rows than block count (grid set to min(256, M))
    K = 1024
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, 1, device="cuda", dtype=torch.float32)
    C = module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    # Allow a slight tolerance for atomic and reduction ordering differences.
    assert torch.allclose(C, C_ref, atol=1e-4), f"Kernel output for multi-block updates differs from reference."

