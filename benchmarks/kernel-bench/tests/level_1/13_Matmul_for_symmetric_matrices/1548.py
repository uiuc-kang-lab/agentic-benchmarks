
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

# Issue 1: Kernel only supports float32 inputs.
def test_input_tensor_type():
    N = 128
    # Create double precision tensors; kernel expects float32, so this should trigger an error.
    A = torch.randn(N, N, dtype=torch.float64, device="cuda")
    B = torch.randn(N, N, dtype=torch.float64, device="cuda")
    mod = build_kernel()
    with pytest.raises(RuntimeError):
        # The kernel’s internal pointer reinterpretation will be incorrect.
        mod.forward(A, B)

# Issue 2: Kernel expects contiguous tensors.
def test_non_contiguous_tensors():
    N = 128
    # Create contiguous float32 tensors then make them non-contiguous by transposing.
    A = torch.randn(N, N, dtype=torch.float32, device="cuda").t()  # not contiguous
    B = torch.randn(N, N, dtype=torch.float32, device="cuda").t()  # not contiguous
    mod = build_kernel()
    # Even though the high-level tensor arithmetic might work if we call .matmul()
    # our kernel (using data_ptr() and assuming row-major contiguous spacing) will produce wrong results.
    C_kernel = mod.forward(A, B)
    torch.cuda.synchronize()
    # Compute correct result from contiguous copies.
    C_ref = torch.matmul(A.contiguous(), B.contiguous())
    # We expect a discrepancy because the kernel does not handle the non-contiguous layout.
    assert not torch.allclose(C_kernel, C_ref, atol=1e-5), \
        "Kernel unexpectedly handled non-contiguous tensors correctly."

# Issue 3: Kernel only supports square matrices.
def test_non_square_inputs():
    # Even if the dimensions allow a legal matrix multiplication in PyTorch,
    # the kernel explicitly checks for square matrices (by comparing A.size(0)==A.size(1))
    # and will throw an error if the input is not square.
    N = 128
    M = 64
    A = torch.randn(N, M, dtype=torch.float32, device="cuda")
    B = torch.randn(M, N, dtype=torch.float32, device="cuda")
    mod = build_kernel()
    with pytest.raises(RuntimeError):
        # The input A is not square so the TORCH_CHECK in the kernel wrapper should trigger.
        mod.forward(A, B)

# Issue 4: No post-kernel launch error checking.
def test_cuda_launch_error():
    # We can simulate a case which may lead to out-of-bound memory access. One approach is to deliberately
    # pass a tensor with an incorrect shape so that the kernel is launched with a grid computed from N
    # that does not match the underlying allocation.
    # For example, we create a valid tensor then slice it so that its reported size is lower than the actual memory layout.
    N = 32
    A = torch.randn(N, N, dtype=torch.float32, device="cuda")
    B = torch.randn(N, N, dtype=torch.float32, device="cuda")
    mod = build_kernel()
    # Slicing A to a smaller size while leaving B intact.
    A_wrong = A[:N-1, :N-1]
    with pytest.raises(RuntimeError):
        mod.forward(A_wrong, B)
