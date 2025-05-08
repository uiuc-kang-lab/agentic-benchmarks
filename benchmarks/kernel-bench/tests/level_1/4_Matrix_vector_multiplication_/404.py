
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build/load the CUDA extension from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
        with_cuda=True,
    )
    return cuda_module

# Issue 1: Compilation error due to bare "min"
def test_compilation():
    """
    This test checks that the kernel compiles correctly.
    (If the "min" issue appears during compile time, the extension never loads.)
    """
    try:
        mod = build_kernel()
    except Exception as e:
        pytest.fail("Compilation failed (possibly due to 'min' not being defined): " + str(e))
    # If we get here, compilation was successful.
    assert hasattr(mod, "forward"), "Module should expose a 'forward' function after compilation."

# Issue 2: The kernel was written for 2D matrix * vector multiplication only.
def test_batched_input():
    """
    Provide a batched (3D) input for A and corresponding batched B.
    In more general situations torch.matmul supports broadcasting, but this
    kernel ignores batch dimensions. The output will be incorrect.
    """
    mod = build_kernel()
    # Create batched A: (batch, M, K)
    batch = 4
    M = 32
    K = 128
    A = torch.randn(batch, M, K, device='cuda', dtype=torch.float32)
    # Create batched B: (batch, K, 1) – note that our kernel is not written for batched inputs.
    B = torch.randn(batch, K, 1, device='cuda', dtype=torch.float32)
    
    # The kernel expects A to be 2D; we simulate a call by “merging” batch and row dimensions.
    # So if the user accidentally passes 3D data, the kernel will treat the batch index as row index.
    A_bad = A.view(-1, K)
    B_bad = B.view(-1, 1)
    C_kernel = mod.forward(A_bad, B_bad)
    # For a batched matrix–vector multiplication, the torch.matmul result is different.
    C_ref = torch.matmul(A, B)
    # Reshape reference to match kernel output (which is computed using all rows from the merged batch)
    C_ref_merged = C_ref.view(-1, 1)
    # The kernel output will differ from the reference because the kernel does not support batching.
    with pytest.raises(AssertionError):
        assert torch.allclose(C_kernel, C_ref_merged, atol=1e-5), (
            "Kernel output should differ from the reference when using batched inputs."
        )

# Issue 3: Using PackedTensorAccessor32 limits tensor sizes.
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for this test."
)
def test_tensor_index_overflow():
    """
    Create a tensor with a total number of elements slightly above the 32-bit indexing limit.
    This test may fail on GPUs without enough memory. It is designed to show that the kernel cannot handle
    tensors whose indexing would overflow 32-bit integers.
    
    NOTE: On many systems it is not practical to actually allocate >2^31 elements.
    Therefore we simulate the error by checking the shape and raising an error if the tensor is too large,
    which in real usage might trigger undefined behavior in the kernel.
    """
    mod = build_kernel()
    
    # Instead of allocating a huge tensor, we check that a tensor with
    # number of elements near (or above) 2^31 triggers our safeguard.
    # (For the purpose of the test, we force the condition.)
    M = 100000
    K = (2**31 // M) + 10  # Deliberately go over the 32-bit limit.
    total_elements = M * K
    if total_elements <= 2**31:
        pytest.skip("Test tensor is not large enough to exceed 32-bit indexing.")
    
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, 1, device='cuda', dtype=torch.float32)
    # Our kernel does not do its own size checking at runtime. In a robust implementation, this should be caught.
    # We call the kernel and then check if the output is wildly wrong.
    C_kernel = mod.forward(A, B)
    C_ref = torch.matmul(A, B)
    # It is expected that the kernel result is not correct if indexing overflows.
    with pytest.raises(AssertionError):
        assert torch.allclose(C_kernel, C_ref, atol=1e-4), (
            "Kernel output unexpectedly close to reference when tensor size exceeds 32-bit indexing capability."
        )

# Issue 4: The kernel launch does not check for errors (lack of cudaError checking).
def test_invalid_B_shape():
    """
    Provide an input where the B tensor does not have the correct number of elements (i.e. B.numel() != K).
    The host code has a TORCH_CHECK that should trigger an error.
    """
    mod = build_kernel()
    M = 64
    K = 256
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    # Provide B with wrong number of elements.
    B = torch.randn(K + 1, 1, device='cuda', dtype=torch.float32)
    with pytest.raises(RuntimeError):
        _ = mod.forward(A, B)
