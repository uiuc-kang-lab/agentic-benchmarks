
import torch
import pytest
from torch.utils.cpp_extension import load
import os

def build_kernel():
    # Make sure we build with full verbosity and appropriate extra flags.
    cuda_module = load(
        name="combined_kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return cuda_module

# Issue 1: Test that using a non float32 tensor (e.g. double) triggers a failure or produces an incorrect result.
def test_input_tensor_type():
    cuda_module = build_kernel()
    N = 512  # using a moderate size for testing

    # Create double precision tensors.
    A = torch.randn(N, N, dtype=torch.double, device='cuda')
    B = torch.randn(N, N, dtype=torch.double, device='cuda')
    A = torch.tril(A)
    B = torch.tril(B)

    # Expect the kernel to misbehave (i.e. not produce the correct result) because it assumes float32.
    # We compare the output with the proper torch.tril(torch.matmul(...)) but with a tolerance that is very small.
    C_kernel = cuda_module.forward(A, B)
    torch.cuda.synchronize()

    C_ref = torch.tril(torch.matmul(A, B))
    # This assertion should fail (or yield a very large difference) because the binary 
    # reinterpretation of double as float32 is not valid.
    diff = (C_kernel - C_ref).abs().max().item()
    assert diff > 1e-3, f"Kernel incorrectly accepted non-float32 input. Max difference {diff}"

# Issue 2: Test that providing non square matrices triggers an error.
def test_input_tensor_dim():
    cuda_module = build_kernel()
    N = 512  # base rows count

    # Create a non-square matrix for B while A remains square.
    A = torch.randn(N, N, dtype=torch.float32, device='cuda')
    A = torch.tril(A)
    # Create B with different column dimension.
    B = torch.randn(N, N + 10, dtype=torch.float32, device='cuda')
    B = torch.tril(B)

    # The kernel code uses A.size(0) for dimension N and then indexes B as if it were N x N.
    # This should cause an out-of-bound access or wrong computation. We expect an error.
    with pytest.raises(RuntimeError):
        C_kernel = cuda_module.forward(A, B)
        torch.cuda.synchronize()

# Issue 3: Lack of CUDA API error checking.
# Although it is not straightforward to force a CUDA API call to fail from Python,
# we simulate the situation by deliberately passing an empty tensor.
def test_empty_input():
    cuda_module = build_kernel()
    # Create an empty tensor for A.
    A = torch.tensor([], dtype=torch.float32, device='cuda').reshape(0,0)
    B = torch.tensor([], dtype=torch.float32, device='cuda').reshape(0,0)
    # As there is no proper dimension for an empty matrix, we expect the kernel to throw an error.
    with pytest.raises(RuntimeError):
        C_kernel = cuda_module.forward(A, B)
        torch.cuda.synchronize()

if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])
