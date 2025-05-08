
import pytest
import torch
from torch.utils.cpp_extension import load

# Utility function to build the CUDA kernel module from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="cuda_kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test case 1: Non-divisible matrix dimensions to expose the boundary handling issue.
def test_non_divisible_dimension():
    # Choose dimensions that are not multiples of TILE_DIM (32). 
    # For example, use M = 33, K = 37, N = 29.
    M, K, N = 33, 37, 29
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)
    my_module = build_kernel()
    
    # Compute output from our custom kernel.
    C_kernel = my_module.forward(A, B)
    torch.cuda.synchronize()
    
    # Compute reference output using torch.matmul.
    C_ref = torch.matmul(A, B)

    # Because of the boundary issue (wrong clamping instead of zero padding),
    # the kernel’s output will differ from the reference.
    # We expect the outputs NOT to be nearly equal.
    is_close = torch.allclose(C_kernel, C_ref, atol=1e-5)
    assert not is_close, (
        "Test failed: The kernel unexpectedly produced results matching torch.matmul "
        "despite non-divisible dimensions. The boundary handling issue might be fixed."
    )

# Test case 2: Passing a tensor of type double (float64) to trigger the type issue.
def test_tensor_type():
    M, K, N = 64, 64, 64
    # Create tensors in double precision.
    A = torch.randn(M, K, device='cuda', dtype=torch.float64)
    B = torch.randn(K, N, device='cuda', dtype=torch.float64)
    my_module = build_kernel()
    
    with pytest.raises(RuntimeError) as excinfo:
        # The kernel expects float32 inputs.
        my_module.forward(A, B)
    assert "must be a CUDA tensor" not in str(excinfo.value), (
        "The error raised should be related to input type, not just CUDA/contiguous check."
    )

# Test case 3: Check that no CUDA errors are caught (or could be caught) after kernel launch.
# This test deliberately uses inputs of mismatched inner dimensions to trigger a kernel launch error.
def test_incorrect_inner_dimension():
    # Mismatch inner dimensions: A: (M, K) and B: (K+1, N)
    M, K, N = 64, 64, 64
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K + 1, N, device='cuda', dtype=torch.float32)
    my_module = build_kernel()
    
    # We expect the kernel call to throw an error (or produce incorrect results) because of dimensional mismatch.
    with pytest.raises(RuntimeError) as excinfo:
        my_module.forward(A, B)
    # In real applications, proper error checking should detect this mismatch rather than silently launching.
    assert "size" in str(excinfo.value) or "incompatible" in str(excinfo.value), (
        "Expected an error message indicating size or dimension mismatch."
    )
    
if __name__ == "__main__":
    pytest.main([__file__])
