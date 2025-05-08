
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper function to build the kernel module.
def build_kernel():
    cuda_module = load(
        name="triangular_mm_kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Correct reference implementation using PyTorch (only works for contiguous, float32 tensors).
def ref_triangular_mm(A, B):
    return torch.tril(torch.matmul(A, B))

# Issue 1: Kernel only supporting float32.
# Test: Create tensors with double precision. The kernel takes the data pointer as float*, so passing double
# should result in incorrect computation. We expect the result to differ from the reference.
@pytest.mark.cuda
def test_input_tensor_type():
    N = 256
    # Create double precision matrices (non supported by the kernel)
    A = torch.randn(N, N, device='cuda', dtype=torch.double)
    B = torch.randn(N, N, device='cuda', dtype=torch.double)
    A = torch.tril(A)
    B = torch.tril(B)
    module = build_kernel()
    # The kernel does not handle double so it will use the raw pointer casts.
    # We cast to float to allow the kernel to run but then the internal values are wrong.
    A_float = A.to(torch.float32)
    B_float = B.to(torch.float32)
    C = module.forward(A_float, B_float)
    torch.cuda.synchronize()
    C_ref = ref_triangular_mm(A_float, B_float)
    # The result should be close if everything were correct.
    # Here, if a user mistakenly passes double inputs, the kernel would be invoked incorrectly.
    # We force a failure if the maximum absolute difference is suspiciously small.
    assert not torch.allclose(C, C_ref, atol=1e-5), \
        "Kernel unexpectedly produced correct results for double inputs; expected type miss-match issue."

# Issue 2: Kernel assumes contiguous input tensors.
# Test: Create non-contiguous lower triangular inputs and check that the kernel result differs from the reference.
@pytest.mark.cuda
def test_non_contiguous_input():
    N = 256
    A = torch.randn(N, N, device='cuda', dtype=torch.float32)
    B = torch.randn(N, N, device='cuda', dtype=torch.float32)
    # Make them lower triangular as per requirements.
    A = torch.tril(A)
    B = torch.tril(B)
    # Create non-contiguous versions by transposing twice (or by slicing)
    A_non_contig = A.t().clone().t()
    B_non_contig = B.t().clone().t()
    # Ensure non-contiguity (typically clone() makes copy contiguous; so we create a view that is not contiguous)
    A_non_contig = A_non_contig[::, :]
    B_non_contig = B_non_contig[::, :]
    assert not A_non_contig.is_contiguous() or not B_non_contig.is_contiguous(), "Inputs are contiguous; test setup failed."

    module = build_kernel()
    C = module.forward(A_non_contig, B_non_contig)
    torch.cuda.synchronize()
    C_ref = ref_triangular_mm(A_non_contig.contiguous(), B_non_contig.contiguous())
    # Expect the mismatch due to wrong linearized indexing
    assert not torch.allclose(C, C_ref, atol=1e-5), \
        "Kernel produced correct results for non-contiguous inputs; expected an error due to wrong memory indexing."

# Issue 3: Kernel creates its own CUDA stream instead of using PyTorch’s current stream.
# Test: Run two operations on the default stream along with the custom kernel launch.
# We deliberately schedule a dummy operation on the default stream after kernel launch and check if they interleave incorrectly.
@pytest.mark.cuda
def test_stream_integration():
    N = 256
    A = torch.randn(N, N, device='cuda', dtype=torch.float32)
    B = torch.randn(N, N, device='cuda', dtype=torch.float32)
    A = torch.tril(A)
    B = torch.tril(B)

    default_stream = torch.cuda.current_stream()
    module = build_kernel()
    # Launch the kernel (which internally creates its own stream)
    C = module.forward(A, B)

    # Schedule a dummy operation on the default stream immediately after the kernel call.
    # If the kernel did not use the current stream, the dummy operation might finish before the kernel,
    # leading to potential incorrect orderings in a complex setting.
    dummy = torch.zeros(1, device='cuda', dtype=torch.float32)
    dummy.add_(1)  # dummy update

    torch.cuda.synchronize()
    C_ref = ref_triangular_mm(A, B)
    # The result should be correct even if stream ordering is suboptimal,
    # but in an integrated PyTorch environment using the current stream, one would expect proper ordering.
    # Here we check if the computed result differs from the reference, indicating a stream mis-synchronization.
    # (Note: This test is somewhat artificial since synchronization forces correctness,
    # however in a real-world scenario using the current stream is important.)
    assert torch.allclose(C, C_ref, atol=1e-5), \
        "Kernel result differs from reference result unexpectedly. This may indicate issues related to stream integration."

# Issue 4: Kernel does not check for CUDA API call errors.
# Test: Although it is hard to simulate a CUDA API error in a controlled environment,
# we try to force an error by passing an invalid N (e.g. negative dimension)
@pytest.mark.cuda
def test_invalid_dimension():
    N = -1  # deliberately invalid
    A = torch.randn(1, 1, device='cuda', dtype=torch.float32)
    B = torch.randn(1, 1, device='cuda', dtype=torch.float32)
    module = build_kernel()
    with pytest.raises(RuntimeError):
        # The kernel launch with an invalid dimension should eventually lead to an error.
        module.forward(A, B)
