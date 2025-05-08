
import torch
import pytest
from torch.utils.cpp_extension import load
import os

def build_kernel():
    # Force rebuild if needed.
    module = load(
        name="test_module",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_input_tensor_type():
    # Issue 1: Passing a tensor with a different type (double) should trigger an error or produce wrong results.
    N = 256
    A = torch.randn(N, N, dtype=torch.double, device='cuda')
    B = torch.randn(N, N, dtype=torch.double, device='cuda')
    
    kernel_module = build_kernel()
    with pytest.raises(RuntimeError):
        # The kernel expects float tensors. An error or exception is expected.
        C = kernel_module.forward(A, B)
        torch.cuda.synchronize()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_non_square_matrices():
    # Issue 2: Providing non-square matrices should not work with a kernel that assumes square matrices.
    # We provide a rectangular input to simulate a more complex situation.
    N, M = 256, 128  # non-square dimensions
    # Create lower-triangular matrices from rectangular matrices.
    A = torch.randn(N, M, dtype=torch.float32, device='cuda')
    B = torch.randn(M, N, dtype=torch.float32, device='cuda')
    
    kernel_module = build_kernel()
    with pytest.raises(RuntimeError):
        # The kernel uses A.size(0) to define the dimension and wrong indexing will occur.
        C = kernel_module.forward(A, B)
        torch.cuda.synchronize()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_missing_algorithm_include(tmp_path):
    # Issue 3: We attempt to compile the kernel and check for compilation errors that might be caused
    # by missing includes, such as <algorithm> for std::min.
    # This test assumes that a compile error would propagate as a RuntimeError.
    try:
        module = build_kernel()
    except Exception as e:
        pytest.fail(f"Kernel compilation failed: {str(e)}")
