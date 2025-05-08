
import torch
import pytest
from torch.utils.cpp_extension import load
import threading

def build_kernel():
    # Build and load the CUDA extension from kernel.cu
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Global constant memory (d_N and d_num_tiles) race conditions.
# We simulate concurrent launches with different matrix sizes.
def test_global_constant_race_condition():
    module = build_kernel()
    
    # Create two matrices with different sizes and launch them concurrently.
    N1 = 128
    N2 = 256
    A1 = torch.randn(N1, N1, device='cuda', dtype=torch.float32).contiguous()
    B1 = torch.randn(N1, N1, device='cuda', dtype=torch.float32).contiguous()
    A2 = torch.randn(N2, N2, device='cuda', dtype=torch.float32).contiguous()
    B2 = torch.randn(N2, N2, device='cuda', dtype=torch.float32).contiguous()
    
    results = {}
    
    def run_kernel(tag, A, B):
        C = module.forward(A, B)
        torch.cuda.synchronize()
        ref = torch.matmul(A, B)
        results[tag] = torch.allclose(C, ref, atol=1e-4)
    
    # Launch two threads that call the kernel concurrently.
    t1 = threading.Thread(target=run_kernel, args=("test1", A1, B1))
    t2 = threading.Thread(target=run_kernel, args=("test2", A2, B2))
    
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    
    # If constant memory race occurs, one or both computations may be wrong.
    assert results["test1"] and results["test2"], "Race condition due to global constant memory variables detected."

# Issue 2: Non-contiguous input tensors.
def test_non_contiguous_inputs():
    module = build_kernel()
    
    N = 256
    A = torch.randn(N, N, device='cuda', dtype=torch.float32)
    B = torch.randn(N, N, device='cuda', dtype=torch.float32)
    # Create non-contiguous versions by transposing
    A_nc = A.t()
    B_nc = B.t()
    # Ensure they are not contiguous.
    assert not A_nc.is_contiguous()
    assert not B_nc.is_contiguous()
    
    C = module.forward(A_nc, B_nc)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A_nc, B_nc)
    assert torch.allclose(C, C_ref, atol=1e-4), "Kernel failed with non-contiguous input tensors."

# Issue 3: Data type limitation to float32.
def test_input_tensor_type():
    module = build_kernel()
    
    N = 128
    # Use double precision input.
    A = torch.randn(N, N, device='cuda', dtype=torch.float64)
    B = torch.randn(N, N, device='cuda', dtype=torch.float64)
    with pytest.raises(RuntimeError):
        # The kernel should reject or produce wrong output when not float32.
        module.forward(A, B)

# Issue 4: Matrix dimensions not a multiple of 4 for vectorized operations.
def test_matrix_dim_not_multiple_of_four():
    module = build_kernel()
    
    # Choose a dimension that is not a multiple of 4.
    N = 67
    A = torch.randn(N, N, device='cuda', dtype=torch.float32).contiguous()
    B = torch.randn(N, N, device='cuda', dtype=torch.float32).contiguous()
    
    C = module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    
    # If misaligned accesses or fallback issues occur, the output will differ.
    assert torch.allclose(C, C_ref, atol=1e-4), "Kernel output is incorrect for matrices with dimensions not multiple of 4."
