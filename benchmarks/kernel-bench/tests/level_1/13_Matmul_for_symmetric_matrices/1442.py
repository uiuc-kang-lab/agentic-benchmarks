
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the extension module from the CUDA source file
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Passing double tensors instead of float32.
def test_input_tensor_type():
    N = 64  # use a modest size for the test
    # Create double precision symmetric matrices
    A = torch.randn(N, N, dtype=torch.float64, device='cuda')
    A = (A + A.T) / 2
    B = torch.randn(N, N, dtype=torch.float64, device='cuda')
    B = (B + B.T) / 2

    module = build_kernel()
    # Expect that the kernel misinterprets double data as float data.
    C_kernel = module.forward(A, B)
    # Compute reference with torch.matmul (which respects dtype)
    C_ref = torch.matmul(A, B)
    # The mismatch in dtype processing should lead to significant differences.
    assert not torch.allclose(C_kernel, C_ref.to(torch.float32), atol=1e-5), \
        "Kernel unexpectedly produced correct result when given double tensors."

# Issue 2: Non-contiguous inputs.
def test_non_contiguous_inputs():
    N = 64  # modest matrix size
    # Create symmetric matrices
    A = torch.randn(N, N, dtype=torch.float32, device='cuda')
    B = torch.randn(N, N, dtype=torch.float32, device='cuda')
    A = (A + A.T) / 2
    B = (B + B.T) / 2

    # Make them non-contiguous by transposing (for a square matrix t() is generally non-contiguous)
    A_noncontig = A.t()
    B_noncontig = B.t()
    assert not A_noncontig.is_contiguous() or not B_noncontig.is_contiguous(), "Inputs are unexpectedly contiguous."

    module = build_kernel()
    C_kernel = module.forward(A_noncontig, B_noncontig)
    # Compute reference; force contiguous conversion on reference to mimic proper matmul behavior.
    C_ref = torch.matmul(A_noncontig.contiguous(), B_noncontig.contiguous())
    # The kernel, however, just uses data_ptr() assuming contiguous data, so the result will differ.
    assert not torch.allclose(C_kernel, C_ref, atol=1e-5), \
        "Kernel unexpectedly handled non-contiguous inputs correctly."

# Issue 3: Non–square matrices.
def test_non_square_matrices():
    # Create inputs with non-square dimensions
    M, K, N = 64, 32, 48
    A = torch.randn(M, K, dtype=torch.float32, device='cuda')
    B = torch.randn(K, N, dtype=torch.float32, device='cuda')
    # Even if the user accidentally provides non–square matrices, our kernel checks that
    # the input matrices are square. So we expect an error.
    module = build_kernel()
    with pytest.raises(RuntimeError):
        _ = module.forward(A, B)

# Issue 4: Lack of error checking after kernel launch.
# To simulate a kernel launch error, we provide an input where N is 0.
# While this is a legal tensor size, it might provoke a situation that the kernel does not check.
def test_empty_matrix():
    N = 0
    A = torch.empty((N, N), dtype=torch.float32, device='cuda')
    B = torch.empty((N, N), dtype=torch.float32, device='cuda')
    module = build_kernel()
    # The kernel will launch but perform no work. We then force a device synchronization to try catching errors.
    C_kernel = module.forward(A, B)
    torch.cuda.synchronize()
    # Compare with reference output (which will also be empty)
    C_ref = torch.matmul(A, B)
    # If there were an underlying error in launching or computing on an edge case, the test could detect it.
    assert C_kernel.numel() == 0 and torch.equal(C_kernel, C_ref), \
        "Kernel did not handle empty inputs as expected."
