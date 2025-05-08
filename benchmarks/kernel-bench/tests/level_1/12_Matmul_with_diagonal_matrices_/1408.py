
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper function to compile/load the CUDA extension.
def build_kernel():
    cuda_module = load(
        name="diag_matmul_extension",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Test that passing a non-float32 tensor (e.g. float64) produces incorrect results.
def test_incorrect_dtype():
    module = build_kernel()
    N = 1024
    M = 64
    # Create tensors of type double (float64)
    A = torch.randn(N, device="cuda", dtype=torch.double)
    B = torch.randn(N, M, device="cuda", dtype=torch.double)
    # Call the CUDA kernel extension (which expects float32 pointers)
    C = module.forward(A, B)
    # Compute the reference output using float32 precision conversion
    A_float = A.float()
    B_float = B.float()
    C_ref = torch.diag(A_float) @ B_float
    # Since the kernel misinterprets the data, the output will be significantly different.
    # The test should catch the error (i.e. output not close to the reference).
    assert not torch.allclose(C, C_ref, atol=1e-5), \
        "Kernel unexpectedly produced correct result for non-float32 inputs!"

# Issue 2: Test that using CPU tensors (which are not compatible with the CUDA kernel) raises an error.
def test_cpu_tensor_error():
    module = build_kernel()
    N = 64
    M = 32
    A = torch.randn(N, device="cpu", dtype=torch.float32)
    B = torch.randn(N, M, device="cpu", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        # The kernel expects CUDA pointers. Using CPU tensors should trigger an error.
        module.forward(A, B)

# Issue 3: Test large dimensions potential overflow.
# It is impractical to allocate huge tensors in a test,
# so we simulate this test by skipping it with an appropriate message.
def test_large_dimensions_overflow(monkeypatch):
    pytest.skip("Skipping overflow test for large dimensions: not practical to allocate such a tensor.")

# Issue 4: Test lack of stream support.
# Since the kernel does not accept a stream parameter, we simulate this test by documenting the limitation.
def test_cuda_stream_support():
    pytest.skip("Skipping CUDA stream support test: the kernel does not implement stream handling.")

if __name__ == "__main__":
    pytest.main([__file__])
