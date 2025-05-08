
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

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_input_tensor_not_float32():
    # Issue 1: Kernel does not handle non-float32 input.
    N = 128
    # Create double tensors (non-float32)
    A = torch.randn(N, N, dtype=torch.float64, device="cuda")
    B = torch.randn(N, N, dtype=torch.float64, device="cuda")
    my_module = build_kernel()
    # Since the underlying kernel reads floats, this should lead to discrepancies.
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A.float(), B.float())
    # The tolerance is loose because the kernel may be computing on garbled data.
    assert not torch.allclose(C, C_ref, atol=1e-5), "Kernel unexpectedly produced correct output with non-float32 input"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_input_tensor_noncontiguous():
    # Issue 2: Kernel assumes contiguous tensors
    N = 128
    A = torch.randn(N, N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, N, device="cuda", dtype=torch.float32)
    # Make A non-contiguous by transposing twice in a way that creates non-contiguity:
    A = A.t()  # Transpose makes it non-contiguous in PyTorch even if sometimes flagged.
    B = B.t()
    assert not A.is_contiguous(), "A should be non-contiguous for this test"
    my_module = build_kernel()
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    # Compute reference result using contiguous clones of A and B.
    C_ref = torch.matmul(A.contiguous(), B.contiguous())
    # The outputs might be incorrect because kernel accesses memory under the assumption of contiguity.
    assert not torch.allclose(C, C_ref, atol=1e-5), "Kernel unexpectedly produced correct output with non-contiguous inputs"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_input_tensor_batched():
    # Issue 3: Kernel is hard-coded for 2D square matrices.
    # Create a simple batched tensor (3D) to simulate a more general use-case.
    batch_size, N = 4, 64
    A = torch.randn(batch_size, N, N, device="cuda", dtype=torch.float32)
    B = torch.randn(batch_size, N, N, device="cuda", dtype=torch.float32)
    my_module = build_kernel()
    # The kernel is not designed for batched inputs so it should either raise an error or produce wrong results.
    with pytest.raises(RuntimeError):
        # Expecting TORCH_CHECK to throw an error because dims != 2.
        _ = my_module.forward(A, B)
