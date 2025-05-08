
import pytest
import torch
from torch.utils.cpp_extension import load
import os

def build_kernel():
    # Ensure the kernel source file exists in the current directory.
    src_file = os.path.join(os.path.dirname(__file__), "kernel.cu")
    cuda_module = load(
        name="diag_matmul_ext",
        sources=[src_file],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: The kernel forces tensor A to be CPU even if provided on GPU.
def test_A_cpu_transfer_behavior():
    # Create A on GPU and B on GPU.
    N = 1024
    M = 512
    A = torch.randn(N, device='cuda', dtype=torch.float32)
    B = torch.randn(N, M, device='cuda', dtype=torch.float32)
    
    module = build_kernel()
    # Even though A is on GPU, the kernel code forces A to be moved to CPU internally.
    # The result computed will be diag(A.cpu()) @ B.
    # This test checks that the behavior is unexpected when providing A as a GPU tensor.
    C = module.forward(A, B)
    torch.cuda.synchronize()
    # Compute reference result with A as provided on GPU.
    C_ref = torch.diag(A) @ B
    # The results will differ because the kernel used A.cpu() and B on GPU.
    # We trigger the issue by asserting that the two are not (or should not be) equal.
    assert not torch.allclose(C, C_ref, atol=1e-5), "Kernel unexpectedly handled GPU tensor A without transferring!"

# Issue 2: The kernel only supports float32.
def test_input_dtype_error():
    N = 512
    M = 256
    # Create A as double tensor.
    A = torch.randn(N, dtype=torch.float64, device='cuda')
    B = torch.randn(N, M, dtype=torch.float64, device='cuda')
    
    module = build_kernel()
    with pytest.raises(RuntimeError):
        # We expect a runtime error or unexpected behavior when providing double tensors.
        module.forward(A, B)
    
# Issue 3: The constant memory buffer has a fixed maximum size.
def test_diag_too_large():
    # Exceed MAX_DIAG_SIZE in the kernel.
    MAX_DIAG_SIZE = 16384
    N = MAX_DIAG_SIZE + 1
    M = 10
    A = torch.randn(N, dtype=torch.float32, device='cuda')
    B = torch.randn(N, M, dtype=torch.float32, device='cuda')
    
    module = build_kernel()
    with pytest.raises(RuntimeError):
        # The kernel contains a TORCH_CHECK that should fail if A.size(0) > MAX_DIAG_SIZE.
        module.forward(A, B)

# Issue 4: No error checking for cudaMemcpyToSymbol or kernel launch.
def test_kernel_launch_error_propagation():
    # We simulate a kernel launch error by providing an invalid tensor for B.
    # Here, instead of a CUDA tensor, we provide a CPU tensor.
    N = 256
    M = 128
    A = torch.randn(N, dtype=torch.float32, device='cuda')
    # B is intentionally left on CPU to trigger device mismatch.
    B = torch.randn(N, M, dtype=torch.float32, device='cpu')
    
    module = build_kernel()
    with pytest.raises(RuntimeError):
        module.forward(A, B)
        
# Issue 5: The kernel assumes B is a CUDA tensor.
def test_B_device_check():
    # Provide a CPU tensor for B to trigger device incompatibility.
    N = 256
    M = 128
    A = torch.randn(N, dtype=torch.float32, device='cuda')
    B = torch.randn(N, M, dtype=torch.float32, device='cpu')
    
    module = build_kernel()
    with pytest.raises(RuntimeError):
        module.forward(A, B)
        
if __name__ == "__main__":
    pytest.main()
