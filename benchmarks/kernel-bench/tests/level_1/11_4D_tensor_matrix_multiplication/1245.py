
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="einsum_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Trigger issue with unsupported data type (double)
def test_invalid_dtype():
    # Build the kernel module
    my_module = build_kernel()
    # Create inputs with double precision which is not supported by the kernel.
    A = torch.randn(2, 4, 4, 8, device="cuda", dtype=torch.float64)
    B = torch.randn(8, 6, device="cuda", dtype=torch.float64)
    # Expect the kernel to either throw an error or produce an incorrect result.
    with pytest.raises(RuntimeError):
        # The TORCH_CHECK in the kernel’s C++ code isn’t checking dtype,
        # so the error might come from a downstream CUDA error.
        C = my_module.forward(A, B)
        torch.cuda.synchronize()

# Test 2: Trigger issue with non-contiguous input tensor
def test_non_contiguous_input():
    my_module = build_kernel()
    # Create a contiguous tensor and then force a non-contiguous one.
    A_contig = torch.randn(2, 4, 4, 8, device="cuda", dtype=torch.float32)
    B = torch.randn(8, 6, device="cuda", dtype=torch.float32)
    # Make A non-contiguous by transposing two dimensions
    A_noncontig = A_contig.transpose(1, 3)
    # Since the kernel assumes a contiguous layout (flattening the first 3 dims),
    # the result will be wrong. We test that the kernel output deviates from the expected einsum.
    C_kernel = my_module.forward(A_noncontig, B)
    C_expected = torch.einsum("bijl,lk->bijk", A_noncontig, B)
    # Allow for potential numerical differences only if the layout was right;
    # here we assert that the kernel result is not matching the reference.
    assert not torch.allclose(C_kernel, C_expected, atol=1e-5), "Kernel unexpectedly handled non-contiguous input!"

# Test 3: Trigger issue with rigid loop unrolling (large L) that might cause resource pressure.
def test_large_L_dimension():
    my_module = build_kernel()
    # Create inputs where L is large.
    # Even if the computation is correct, the kernel might perform poorly or even crash.
    BATCH, I, J, L, K = 2, 8, 8, 2048, 16
    A = torch.randn(BATCH, I, J, L, device="cuda", dtype=torch.float32)
    B = torch.randn(L, K, device="cuda", dtype=torch.float32)
    try:
        C = my_module.forward(A, B)
        torch.cuda.synchronize()
        # As a minimal correctness check, compare with torch.einsum.
        C_expected = torch.einsum("bijl,lk->bijk", A, B)
        # Depending on the unrolling, the kernel may return wrong results.
        if not torch.allclose(C, C_expected, atol=1e-3):
            raise RuntimeError("Kernel produced incorrect result for large L.")
    except RuntimeError as e:
        pytest.fail(f"Kernel failed for large L dimension: {str(e)}")

# Test 4: Check if kernel launch errors are properly (or improperly) checked.
def test_kernel_launch_error_checking():
    my_module = build_kernel()
    # Provide deliberately mismatched dimensions so that the kernel's assumptions break.
    # For example, let B have a wrong shape relative to A.
    A = torch.randn(2, 4, 4, 10, device="cuda", dtype=torch.float32)
    B = torch.randn(9, 6, device="cuda", dtype=torch.float32)  # B.size(0) != A.size(3)
    with pytest.raises(RuntimeError):
        C = my_module.forward(A, B)
        torch.cuda.synchronize()
