
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="triangular_mm",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Non–contiguous input tensors.
# The kernel directly indexes memory under the assumption of contiguity.
def test_non_contiguous_input():
    N = 128
    # Create a contiguous lower-triangular matrix.
    A = torch.tril(torch.randn(N, N, device="cuda", dtype=torch.float32))
    B = torch.tril(torch.randn(N, N, device="cuda", dtype=torch.float32))
    # Make them non–contiguous by transposing.
    A_noncontig = A.t()
    B_noncontig = B.t()
    # Ensure they are really non–contiguous.
    assert not A_noncontig.is_contiguous()
    assert not B_noncontig.is_contiguous()
    kernel = build_kernel()
    C_kernel = kernel.forward(A_noncontig, B_noncontig)
    # Reference computed using torch cannot use non-contiguous tensors directly in this case,
    # so we force contiguous copies for the reference.
    C_ref = torch.tril(torch.matmul(A_noncontig.contiguous(), B_noncontig.contiguous()))
    # Since the kernel did not check for non–contiguity it will use the wrong memory layout.
    # Therefore, we expect the result to differ significantly from the reference.
    assert not torch.allclose(C_kernel, C_ref, atol=1e-5), "Kernel should produce wrong output on non–contiguous inputs!"

# Issue 2: Kernel only supports float32 input.
def test_wrong_dtype():
    N = 64
    A = torch.tril(torch.randn(N, N, device="cuda", dtype=torch.float64))
    B = torch.tril(torch.randn(N, N, device="cuda", dtype=torch.float64))
    kernel = build_kernel()
    with pytest.raises(RuntimeError):
        # When calling the kernel with float64, the pointer conversion (data_ptr<float>())
        # is invalid. An error should be thrown.
        _ = kernel.forward(A, B)

# Issue 3: Use of improper loop unrolling assumption.
# While this may not crash the kernel, it can lead to subtle performance or correctness issues
# when the loop bounds are non–constant. We simulate this by testing matrices where the number
# of multiply–accumulate iterations varies significantly, and then compare with the expected result.
def test_variable_loop_bounds():
    N = 128
    A = torch.tril(torch.randn(N, N, device="cuda", dtype=torch.float32))
    B = torch.tril(torch.randn(N, N, device="cuda", dtype=torch.float32))
    kernel = build_kernel()
    C_kernel = kernel.forward(A, B)
    # Compute the reference using torch.matmul and then lower-triangularize the result.
    C_ref = torch.tril(torch.matmul(A, B))
    # Even if the loop unrolling is suboptimal, the result (if computed correctly)
    # should match the reference.
    assert torch.allclose(C_kernel, C_ref, atol=1e-3), \
        f"Kernel output does not match reference! Max difference: {(C_kernel - C_ref).abs().max()}"

if __name__ == "__main__":
    pytest.main([__file__])
