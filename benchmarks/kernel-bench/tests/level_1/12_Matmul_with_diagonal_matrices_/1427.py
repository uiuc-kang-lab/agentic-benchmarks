
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="diag_matmul_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Trigger the bug due to improper broadcast method.
# If every thread were incorrectly loading the diagonal value instead of one doing a load and then broadcasting,
# the results may be incorrect when diagonal values differ significantly. We create a scenario where each diagonal element is unique.
def test_broadcast_logic():
    device = "cuda"
    N = 128
    M = 256  # divisible by 4 so vectorized branch is taken
    # Make diagonal elements increasing so that errors in broadcasting would change rows completely.
    A = torch.arange(1, N + 1, dtype=torch.float32, device=device)
    # Create tensor B with random values.
    B = torch.randn(N, M, dtype=torch.float32, device=device)
    
    # Expected: C[i, :] should equal A[i] * B[i, :]
    ref = A.view(-1, 1) * B
    module = build_kernel()
    C = module.forward(A, B)
    torch.cuda.synchronize()
    # If the broadcast is wrong, then some rows may get a wrong diag value.
    assert torch.allclose(C, ref, atol=1e-5), f"Broadcast issue: max difference = {(C - ref).abs().max().item()}"

# Test 2: Trigger type-check issue. Passing a tensor of type double should cause a kernel failure or produce wrong results.
def test_wrong_dtype():
    device = "cuda"
    N = 128
    M = 128
    # Use double precision for A and B.
    A = torch.randn(N, dtype=torch.double, device=device)
    B = torch.randn(N, M, dtype=torch.double, device=device)
    module = build_kernel()
    with pytest.raises(RuntimeError):
        # Our kernel assumes float pointer casting. Either an error is thrown,
        # or (if not caught) the results will be wrong.
        C = module.forward(A, B)
        torch.cuda.synchronize()

# Test 3: Test the vectorized path misalignment issue.
# We create a tensor whose underlying storage is not appropriately aligned for vectorized loads.
# We do so by creating a padded tensor and then using narrow slicing to force a misalignment.
def test_misaligned_memory():
    device = "cuda"
    N = 128
    M = 260   # choose M divisible by 4 so that vectorized path is entered.
    # Create a larger tensor and then slice it to force a non-ideal alignment.
    large_B = torch.randn(N, M + 1, dtype=torch.float32, device=device)
    B = large_B[:, 1:]  # This slice might not be aligned to 16 bytes.
    A = torch.randn(N, dtype=torch.float32, device=device)
    
    # Compute reference result using PyTorch.
    ref = A.view(-1, 1) * B
    module = build_kernel()
    C = module.forward(A, B)
    torch.cuda.synchronize()
    # The result might be wrong because the vectorized load in the kernel is based on reinterpret_cast,
    # and misalignment could lead to incorrect accesses.
    assert torch.allclose(C, ref, atol=1e-5), f"Misaligned memory issue detected: max difference = {(C - ref).abs().max().item()}"

# Test 4: Test the scalar fallback path.
# Here we choose M that is not divisible by 4 so that the kernel uses the scalar processing function.
def test_scalar_path():
    device = "cuda"
    N = 128
    M = 257  # Not divisible by 4
    A = torch.randn(N, dtype=torch.float32, device=device)
    B = torch.randn(N, M, dtype=torch.float32, device=device)
    
    ref = A.view(-1, 1) * B
    module = build_kernel()
    C = module.forward(A, B)
    torch.cuda.synchronize()
    assert torch.allclose(C, ref, atol=1e-5), f"Scalar path issue: max difference = {(C - ref).abs().max().item()}"
    
if __name__ == "__main__":
    pytest.main([__file__])
