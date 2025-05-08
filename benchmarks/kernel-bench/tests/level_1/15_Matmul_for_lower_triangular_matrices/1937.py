
import pytest
import torch
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

# Test 1: Passing non‑float32 tensors to trigger the hard-coded float assumption.
def test_input_tensor_type():
    N = 128
    A = torch.randn(N, N, device='cuda', dtype=torch.double)  # double instead of float
    B = torch.randn(N, N, device='cuda', dtype=torch.double)
    my_module = build_kernel()
    with pytest.raises(RuntimeError):
        # Expect the kernel to fail or produce an error because of dtype mismatch.
        C = my_module.forward(A, B)
        torch.cuda.synchronize()

# Test 2: Passing non‑square matrices to trigger the limitation for square inputs.
def test_non_square_matrix():
    # Create a rectangular matrix (e.g., 128 x 256).
    A = torch.randn(128, 256, device='cuda', dtype=torch.float32)
    B = torch.randn(128, 256, device='cuda', dtype=torch.float32)
    my_module = build_kernel()
    with pytest.raises(RuntimeError):
        # The extension should assert that the tensor is square.
        C = my_module.forward(A, B)
        torch.cuda.synchronize()

# Test 3: Passing batched (3D) input to trigger the limitation regarding only 2D matrices.
def test_batched_input():
    # Create a batched lower-triangular matrix (batch, N, N)
    N = 64
    A = torch.randn(4, N, N, device='cuda', dtype=torch.float32)
    B = torch.randn(4, N, N, device='cuda', dtype=torch.float32)
    # Even if we extract one batch, the intention here is to show that multiple batches are not supported.
    my_module = build_kernel()
    with pytest.raises(RuntimeError):
        # The kernel and its forward interface check that the tensors are 2D.
        C = my_module.forward(A, B)
        torch.cuda.synchronize()
        
# Test 4: (Optional) A sanity test for the fixed-block configuration.
# Although the kernel may compute the correct result for valid input,
# the design assumes a fixed thread block shape. Here we test with a size that stresses the tiling.
def test_fixed_tile_dimensions():
    N = 256  # Use a size that is not a multiple of the tile sizes to check the boundaries.
    A = torch.randn(N, N, device='cuda', dtype=torch.float32)
    B = torch.randn(N, N, device='cuda', dtype=torch.float32)
    # Ensure lower triangular structure.
    A = torch.tril(A)
    B = torch.tril(B)
    my_module = build_kernel()
    # Even though the kernel is rigid, it should compute the lower triangular multiplication correctly.
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    # Compute reference result in PyTorch
    C_ref = torch.tril(torch.matmul(A, B))
    assert torch.allclose(C, C_ref, atol=1e-5), "Kernel result does not match reference output."
