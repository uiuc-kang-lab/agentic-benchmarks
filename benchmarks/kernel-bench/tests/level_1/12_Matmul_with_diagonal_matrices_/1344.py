
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="diag_mul_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,  # Set true for debugging build
    )
    return cuda_module

# Issue 1: Kernel is hard-coded to float32. Passing double tensors should produce wrong results.
def test_wrong_dtype():
    my_module = build_kernel()
    N = 1024
    M = 1024
    # Create double tensors on CUDA. The kernel expects float so the behavior is undefined.
    A = torch.randn(N, device='cuda', dtype=torch.double)
    B = torch.randn(N, M, device='cuda', dtype=torch.double)
    # Compute reference on CPU after casting to float, so we know what proper result should be.
    A_float = A.float()
    B_float = B.float()
    C_ref = torch.diag(A_float) @ B_float

    # Call the kernel extension. It does not check dtype and uses A.data_ptr<float>().
    C = my_module.forward(A, B)

    # The output is likely not matching because of the type mismatch.
    # We check that the values are far off the expected result.
    diff = (C - C_ref).abs().max().item()
    assert diff > 1e-3, f"Kernel incorrectly accepted double inputs! Max diff: {diff}"

# Issue 2: Kernel does not check that input tensors are on CUDA.
def test_wrong_device():
    my_module = build_kernel()
    N = 512
    M = 512
    # Create tensors on CPU (they are float32 and shaped correctly)
    A = torch.randn(N, device='cpu', dtype=torch.float32)
    B = torch.randn(N, M, device='cpu', dtype=torch.float32)
    with pytest.raises(RuntimeError):
        # This should raise an error because the kernel launch expects CUDA tensors.
        C = my_module.forward(A, B)

# Issue 3: Kernel only supports a 1D vector for A and a 2D matrix for B.
def test_batched_input():
    my_module = build_kernel()
    batch = 4
    N = 256
    M = 256
    # Create batched tensors. In a batched scenario, A might be (batch, N) and B might be (batch, N, M).
    A = torch.randn(batch, N, device='cuda', dtype=torch.float32)
    B = torch.randn(batch, N, M, device='cuda', dtype=torch.float32)
    # The kernel's forward checks require A.dim()==1 and B.dim()==2.
    with pytest.raises(RuntimeError) as excinfo:
        C = my_module.forward(A, B)
    expected_msg = "A must be a 1D tensor"
    assert expected_msg in str(excinfo.value), f"Expected error message containing '{expected_msg}' but got '{excinfo.value}'"
