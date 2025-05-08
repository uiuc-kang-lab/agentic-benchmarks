
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="triangular_mm_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

def compute_reference(A, B):
    # Compute full matmul and then extract the lower triangular part.
    return torch.tril(torch.matmul(A, B))

def test_data_type_limitation():
    # Issue 1: Passing a double tensor should trigger an error or produce incorrect result.
    my_module = build_kernel()
    N = 128
    # Create double precision tensors
    A = torch.randn(N, N, dtype=torch.double, device='cuda')
    B = torch.randn(N, N, dtype=torch.double, device='cuda')
    # The kernel expects float input but we are supplying double.
    # We expect the results to be off or a runtime error.
    with pytest.raises(RuntimeError):
        # If the kernel does not check types explicitly, the wrong interpretation of data can lead to a crash.
        C = my_module.forward(A, B)
        torch.cuda.synchronize()

def test_contiguity_assumption():
    # Issue 2: The kernel assumes contiguous memory.
    my_module = build_kernel()
    N = 128
    # Create contiguous lower triangular matrices.
    A = torch.tril(torch.randn(N, N, device="cuda", dtype=torch.float32))
    B = torch.tril(torch.randn(N, N, device="cuda", dtype=torch.float32))
    # Make them non-contiguous by taking a transpose (which for square matrices generally produces noncontiguous layout)
    A_nc = A.t()
    B_nc = B.t()
    # The lower triangular property is broken by a simple transpose,
    # so enforce it again (note: the resulting tensor remains noncontiguous).
    A_nc = torch.tril(A_nc)
    B_nc = torch.tril(B_nc)
    C = my_module.forward(A_nc, B_nc)
    torch.cuda.synchronize()
    # We use contiguous copies to compute the reference.
    C_ref = compute_reference(A_nc.contiguous(), B_nc.contiguous())
    # Because the kernel did not account for non-contiguous memory layouts,
    # the result will likely differ from the correct reference.
    assert not torch.allclose(C, C_ref, atol=1e-5), (
        f"Kernel unexpectedly handled non-contiguous inputs correctly. "
        f"Max difference: {(C-C_ref).abs().max()}"
    )

def test_batched_input_not_supported():
    # Issue 3: The kernel does not support batched inputs.
    my_module = build_kernel()
    N = 128
    # Create a batched tensor (3D tensor) which should not be accepted.
    A = torch.tril(torch.randn(2, N, N, device="cuda", dtype=torch.float32))
    B = torch.tril(torch.randn(2, N, N, device="cuda", dtype=torch.float32))
    with pytest.raises(RuntimeError):
        # Since the C++ wrapper checks that A.dim() == 2, passing a 3D tensor should raise an error.
        C = my_module.forward(A, B)
        torch.cuda.synchronize()

def test_unroll_behavior():
    # Issue 4: The use of "#pragma unroll" on a loop with a dynamic iteration count.
    # This issue is more about performance/potential miscompilation than a correctness error.
    # We simulate a scenario with a large dynamic range in the loop bound.
    my_module = build_kernel()
    N = 1024  # Larger dimension to force larger loop iterations (when row-col is large)
    A = torch.tril(torch.randn(N, N, device="cuda", dtype=torch.float32))
    B = torch.tril(torch.randn(N, N, device="cuda", dtype=torch.float32))
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = compute_reference(A, B)
    # Even if the kernel produces a “correct” result for some values,
    # the heavy dependence on dynamic loop lengths with unrolling may affect performance or numerical stability.
    # Here we only perform a basic check and warn if the results are too off.
    assert torch.allclose(C, C_ref, atol=1e-4) == False, (
        "Kernel appears to handle dynamic unrolling unexpectedly well; "
        "this test expects that large loop bounds may cause numerical differences or performance issues."
    )

if __name__ == "__main__":
    pytest.main([__file__])
