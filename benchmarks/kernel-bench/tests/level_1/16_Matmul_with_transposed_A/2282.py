
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper to build and load the CUDA extension from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 2: Test that non-float32 inputs trigger the expected error.
def test_input_tensor_type():
    my_module = build_kernel()
    N = 128
    # Create double precision tensors (instead of float32).
    A = torch.randn(256, 64, dtype=torch.float64, device="cuda")
    B = torch.randn(256, N, dtype=torch.float64, device="cuda")
    with pytest.raises(RuntimeError, match="Input A must be float32"):
        _ = my_module.forward(A, B)

# Issue 3: Test with non-contiguous inputs.
def test_non_contiguous_input():
    my_module = build_kernel()
    # Create contiguous inputs.
    K, M, N = 128, 64, 256
    A = torch.randn(K, M, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    # Make A non-contiguous by transposing it (remember A is expected in shape (K, M)).
    A_noncontig = A.t()  # Now shape is (M, K) and non-contiguous.
    # Note: This does not follow the expected layout.
    # We expect the kernel's internal indexing (which assumes contiguous layout) to produce a wrong result.
    C_kernel = my_module.forward(A_noncontig, B)
    # Compute the “expected” result using PyTorch's matmul on the same non-contiguous tensor.
    # But note: the Python model does A.T * B. Here if we pass a non-contiguous A, then A.T gives back A (if A was original transposed)
    # So we simulate the intended operation manually:
    C_expected = torch.matmul(A_noncontig.t(), B)
    # They likely will differ because kernel indexing does not account for non-contiguity.
    assert not torch.allclose(C_kernel, C_expected, atol=1e-5), (
        "Kernel output should differ due to non-contiguous input assumptions."
    )

# Issue 4: Test that the kernel only works for A.T * B.
def test_wrong_shape_input():
    my_module = build_kernel()
    # Instead of passing A in expected shape (K, M), pass A in shape (M, K) so that no transposition is needed.
    M, K, N = 64, 128, 256
    # Note: This is the opposite of the expected layout.
    A = torch.randn(M, K, device="cuda", dtype=torch.float32")  # incorrect shape for expected behavior.
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    # The intended computation from Python is torch.matmul(A.T, B) but if A has shape (M, K),
    # then A.T has shape (K, M) and the multiplication result will be of shape (M, N).
    # Because our kernel always treats the first tensor as (K, M) stored in row-major,
    # the kernel will interpret the data of A incorrectly.
    C_kernel = my_module.forward(A, B)
    # Compute the expected result using the Python model: first transpose A then matmul.
    C_expected = torch.matmul(A.t(), B)
    # Because of the wrong input shape, we expect the kernel output to be different.
    assert not torch.allclose(C_kernel, C_expected, atol=1e-5), (
        "Kernel output should differ when input A is not in the expected (K, M) layout."
    )

# Issue 5: Test dimension mismatches and error reporting.
def test_dimension_mismatch():
    my_module = build_kernel()
    # Create tensors with mismatched dimensions.
    A = torch.randn(128, 64, device="cuda", dtype=torch.float32)
    # B has an incompatible first dimension (should be 128 to match A.size(0)).
    B = torch.randn(120, 256, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="Dimension mismatch"):
        _ = my_module.forward(A, B)

if __name__ == "__main__":
    pytest.main([__file__])
