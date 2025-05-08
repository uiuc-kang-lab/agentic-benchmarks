
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Test case 1: Trigger issue with unsupported tensor type (float16)
def test_unsupported_dtype_half():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    M = 128
    K = 256
    # Create half precision tensors
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, 1, device="cuda", dtype=torch.float16)
    module = build_kernel()
    with pytest.raises(RuntimeError):
        # Expect RuntimeError because half is not in AT_DISPATCH_FLOATING_TYPES
        C = module.forward(A, B)
        torch.cuda.synchronize()

# Test case 2: Trigger issue with unsupported batched input dimensions
def test_batched_input_not_supported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    # Create batched input: A has shape (batch, M, K), B has shape (batch, K, 1)
    batch = 4
    M = 64
    K = 128
    A = torch.randn(batch, M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(batch, K, 1, device="cuda", dtype=torch.float32)
    module = build_kernel()
    with pytest.raises(IndexError):
        # Since the kernel expects a 2D A (and 1D/2D B) it should error out
        # when trying to access dimensions that don't exist
        C = module.forward(A, B)
        torch.cuda.synchronize()

# Test case 3: Trigger potential misaligned load scenario
def test_non_aligned_input():
    """
    This test case attempts to trigger misaligned memory access.
    It creates a tensor with an offset using as_strided. Although the wrapper
    calls contiguous() (which usually results in aligned memory), this simulates
    a scenario where the kernel might encounter misaligned pointers.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    M = 64
    K = 130  # choose a K that is not a multiple of 4 to force remainder iterations
    # Create a larger tensor and then get a sub-tensor that is potentially misaligned.
    base = torch.randn(M, K+1, device="cuda", dtype=torch.float32)
    # Create a sub-tensor that is not pointing to the beginning of a memory allocation
    A = base[:, 1:].clone()  # clone to avoid contiguous guarantee from as_strided on a subview
    # Ensure A is non-contiguous before our clone (simulate non-aligned memory)
    A = A.as_strided((M, K), (A.stride(0), A.stride(1)))
    
    # For vector B, we do similar manipulation.
    base_B = torch.randn(K+1, device="cuda", dtype=torch.float32)
    B = base_B[1:].clone().view(K, 1)
    module = build_kernel()
    
    # This test does not necessarily expect a crash but will compare against torch.matmul.
    # If misaligned accesses cause incorrect computation the result will differ.
    C = module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    # We use a loose tolerance because misaligned loads might slightly affect performance but must be correct.
    assert torch.allclose(C, C_ref, atol=1e-5), f"Kernel output differs from torch.matmul output! Max diff: {(C-C_ref).abs().max()}"
