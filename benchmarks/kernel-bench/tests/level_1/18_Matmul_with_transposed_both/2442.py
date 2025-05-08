
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the CUDA extension from kernel.cu.
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 2: Non‐contiguous memory view causing incorrect indexing.
def test_non_contiguous_inputs():
    my_module = build_kernel()
    # Create contiguous tensors for A and B.
    M = 1024
    K = 4096
    N = 2048
    A = torch.randn(K, M, device='cuda', dtype=torch.float32)
    B = torch.randn(N, K, device='cuda', dtype=torch.float32)
    # Create non-contiguous versions using as_strided.
    # For a contiguous tensor of shape (K, M) the strides are (M, 1); here we force new strides.
    A_nc = A.as_strided(size=A.size(), stride=(1, A.size(0)))
    B_nc = B.as_strided(size=B.size(), stride=(1, B.size(0)))
    # Call the kernel with non-contiguous inputs.
    C = my_module.forward(A_nc, B_nc)
    torch.cuda.synchronize()
    # Compute reference from contiguous inputs.
    C_ref = torch.matmul(A.t(), B.t())
    # The kernel computes using a simple indexing formula so the result is expected to differ.
    # (There is no runtime error – the error is in the computed result.)
    assert not torch.allclose(C, C_ref, atol=1e-5), \
        "Test failed: non-contiguous inputs did not trigger an indexing error."

# Issue 3: Non-floating point types not supported by the dispatch.
def test_non_floating_point():
    my_module = build_kernel()
    # Use small sizes for simplicity.
    M = 32
    K = 64
    N = 16
    # Create integer tensors instead of floating-point.
    A_int = torch.randint(0, 10, (K, M), device='cuda', dtype=torch.int32)
    B_int = torch.randint(0, 10, (N, K), device='cuda', dtype=torch.int32)
    with pytest.raises(RuntimeError):
        # Expecting the kernel to throw an error because the dispatch macro does not handle int.
        _ = my_module.forward(A_int, B_int)

# Issue 1: Misleading “128-bit” load optimization.
# While the kernel employs __ldg() for each scalar element, it does not actually use 128-bit (vectorized) loads.
# There is no runtime error for this issue but one may detect performance or correctness problems
# when the inputs are correctly aligned. To simulate one aspect of the alignment dependency,
# we test for correctness when we feed inputs that (although contiguous)
# are not guaranteed to be 128-bit aligned.
def test_alignment_issue():
    my_module = build_kernel()
    # Use dimensions that are multiples of the tile sizes.
    M = 1024
    K = 4096
    N = 2048
    # Create contiguous tensors.
    A = torch.randn(K, M, device='cuda', dtype=torch.float32)
    B = torch.randn(N, K, device='cuda', dtype=torch.float32)
    # Force misalignment by adding an extra byte offset.
    # In PyTorch it is not trivial to force pointer misalignment directly.
    # One workaround is to create a larger buffer and then use as_strided over a misaligned sub-tensor.
    A_buf = torch.empty(K * M + 1, device='cuda', dtype=torch.float32)
    B_buf = torch.empty(N * K + 1, device='cuda', dtype=torch.float32)
    # Copy our contiguous data into these buffers.
    A_buf[1:].copy_(A.view(-1))
    B_buf[1:].copy_(B.view(-1))
    A_misaligned = A_buf[1:].view(K, M)
    B_misaligned = B_buf[1:].view(N, K)
    # Run the kernel using misaligned data. (The kernel uses __ldg but does not force 128-bit loads.)
    C = my_module.forward(A_misaligned, B_misaligned)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A_misaligned.t(), B_misaligned.t())
    # Since the kernel’s loads are not truly vectorized 128-bit loads,
    # these misaligned inputs may produce an incorrect result.
    # (The test expects the results to be wrong – triggering the “issue”.)
    assert not torch.allclose(C, C_ref, atol=1e-5), \
        "Test failed: misaligned inputs did not trigger a difference in output."

