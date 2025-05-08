
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

# Test 1: Trigger type issue by passing double tensors.
def test_input_tensor_type():
    # Create tensors of type float64.
    M, K, N = 64, 128, 32
    A = torch.randn(M, K, device="cuda", dtype=torch.float64)
    # B is expected to be in transposed form: shape (N, K)
    B = torch.randn(N, K, device="cuda", dtype=torch.float64)
    
    my_kernel = build_kernel()
    # Since the kernel assumes float32, the results will be incorrect.
    C = my_kernel.forward(A, B)
    torch.cuda.synchronize()
    
    # Compute reference using the intended operation (note: using B.T to mimic the Python code behavior).
    # However, with double values, the kernel’s float32 interpretation will produce wrong results.
    C_ref = torch.matmul(A.to(torch.float32), B.to(torch.float32).T)
    
    # The test should detect the discrepancy.
    assert not torch.allclose(C, C_ref, atol=1e-3), (
        "Kernel did not trigger type issue: using double tensors produced a result matching float32 reference."
    )

# Test 2: Trigger non-contiguous tensor issue.
def test_non_contiguous_input():
    # Create contiguous tensor and then create a non-contiguous view.
    M, K, N = 64, 128, 32
    A_contig = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B_contig = torch.randn(N, K, device="cuda", dtype=torch.float32)
    
    # Create non-contiguous views by transposing a contiguous tensor.
    A = A_contig.T  # Now shape is (K, M) and non-contiguous.
    B = B_contig  # B remains contiguous.
    
    my_kernel = build_kernel()
    with pytest.raises(RuntimeError, match="Inputs must be contiguous"):
        # This should trigger the TORCH_CHECK for contiguous inputs.
        my_kernel.forward(A, B)
    
# Test 3: Trigger incorrect B shape issue.
def test_incorrect_B_shape():
    # Here we purposely provide B in conventional shape of (K, N) instead of expected (N, K).
    M, K, N = 64, 128, 32
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    # Provide B with shape (K, N) instead of (N, K)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    
    my_kernel = build_kernel()
    # The kernel will interpret B as having shape (N, K). When B is (K, N),
    # the multiplication will compute incorrect dot products.
    C = my_kernel.forward(A, B)
    torch.cuda.synchronize()
    
    # Compute the intended correct result (as per our Python model, using B.T).
    # The expected behavior per the Python model is torch.matmul(A, B.T), which means the actual kernel should be invoked with B of shape (N, K).
    # Here, deliberately using B in conventional form should yield a result different from torch.matmul(A, B.T).
    C_ref = torch.matmul(A, B.T)
    assert not torch.allclose(C, C_ref, atol=1e-5), (
        "Kernel did not trigger incorrect B shape issue: results match when B is provided in conventional shape."
    )

if __name__ == "__main__":
    pytest.main([__file__])
