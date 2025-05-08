
import torch
import pytest
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

# Issue 1: Test for non-contiguous input tensors.
def test_non_contiguous_tensors():
    # Create contiguous tensors first.
    A = torch.randn(128, 4096, device="cuda", dtype=torch.float32)
    B = torch.randn(128, 4096, device="cuda", dtype=torch.float32)
    # Make them non-contiguous by transposing twice (or slicing)
    A_noncontig = A.t().contiguous().t()[::1, :]
    B_noncontig = B.t().contiguous().t()[::1, :]
    assert not A_noncontig.is_contiguous(), "Test setup error: tensor A is contiguous."
    my_module = build_kernel()
    # Even if the host function does not check for non-contiguity,
    # our kernel will read memory incorrectly.
    output = my_module.forward(A_noncontig, B_noncontig)
    # We expect the result to differ from what PyTorch computes if memory layout were correct.
    expected = torch.mean(1 - torch.nn.functional.cosine_similarity(A_noncontig, B_noncontig, dim=1))
    # This test is set to fail if the kernel is used with non-contiguous tensors.
    assert not torch.allclose(output, expected, atol=1e-5), (
        "Kernel did not trigger an error with non-contiguous inputs; its results appear unexpectedly correct."
    )

# Issue 2: Test behavior when inner dimension D is significantly smaller than the block size.
def test_small_inner_dimension():
    # Using a very small inner dimension
    D_small = 16  # much smaller than the fixed 512 threads per block.
    A = torch.randn(128, D_small, device="cuda", dtype=torch.float32)
    B = torch.randn(128, D_small, device="cuda", dtype=torch.float32)
    my_module = build_kernel()
    output = my_module.forward(A, B)
    expected = torch.mean(1 - torch.nn.functional.cosine_similarity(A, B, dim=1))
    # While the kernel may yield a correct numerical result, the inefficiency
    # (or potential underutilization) is the problem we want to catch.
    # Here we check if there is a difference beyond an acceptable tolerance.
    assert torch.allclose(output, expected, atol=1e-5), (
        "Kernel output for small inner dimension does not match the expected value."
    )
    # A warning in profiling would be preferred, however in this test we simply note that
    # an inefficient configuration was used.

# Issue 3: Test potential numerical instability due to the use of atomicAdd.
def test_numerical_stability():
    # Create a scenario with many rows to exacerbate accumulation rounding errors.
    N = 8192   # a large number of rows so that many atomicAdds are performed.
    D = 4096
    A = torch.randn(N, D, device="cuda", dtype=torch.float32)
    B = torch.randn(N, D, device="cuda", dtype=torch.float32)
    my_module = build_kernel()
    output = my_module.forward(A, B)
    expected = torch.mean(1 - torch.nn.functional.cosine_similarity(A, B, dim=1))
    # For many accumulations there could be a notable difference even if within tolerance;
    # here we check that the discrepancy is at least marginally larger than a very tight tolerance.
    diff = torch.abs(output - expected).item()
    assert diff > 1e-6, (
        f"Expected a difference (due to potential atomicAdd numerical instability) but got diff {diff}"
    )
