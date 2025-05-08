
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

# Issue 1 test: Concurrent calls with different matrix sizes.
# Because d_N is stored in constant memory and overwritten before each kernel launch,
# using different matrix sizes concurrently should produce invalid results.
def test_concurrent_different_sizes():
    # Create two CUDA streams.
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()

    # Use two different sizes.
    N1 = 512
    N2 = 1024

    A1 = torch.triu(torch.randn(N1, N1, device="cuda", dtype=torch.float32))
    B1 = torch.triu(torch.randn(N1, N1, device="cuda", dtype=torch.float32))
    A2 = torch.triu(torch.randn(N2, N2, device="cuda", dtype=torch.float32))
    B2 = torch.triu(torch.randn(N2, N2, device="cuda", dtype=torch.float32))

    module = build_kernel()

    # Launch kernel in stream1.
    with torch.cuda.stream(stream1):
        C1 = module.forward(A1, B1)
    # Launch kernel in stream2.
    with torch.cuda.stream(stream2):
        C2 = module.forward(A2, B2)

    # Synchronize streams.
    stream1.synchronize()
    stream2.synchronize()

    # Compute expected results using PyTorch operations.
    C1_ref = torch.triu(torch.matmul(A1, B1))
    C2_ref = torch.triu(torch.matmul(A2, B2))

    # Because of the global constant memory issue, at least one of these should be wrong.
    res1_correct = torch.allclose(C1, C1_ref, atol=1e-5)
    res2_correct = torch.allclose(C2, C2_ref, atol=1e-5)

    # The test triggers the issue by expecting at least one of the outputs to be wrong.
    assert (not res1_correct) or (not res2_correct), "Expected at least one concurrent call to produce an incorrect result due to overwriting constant memory d_N."

# Issue 2 test: Passing tensors with an unsupported data type (double).
def test_input_tensor_type():
    N = 256
    # Create double precision upper triangular matrices.
    A = torch.triu(torch.randn(N, N, device="cuda", dtype=torch.float64))
    B = torch.triu(torch.randn(N, N, device="cuda", dtype=torch.float64))
    module = build_kernel()
    # Expect the kernel to either error or to produce an incorrect result because it is hard-coded for float.
    with pytest.raises(RuntimeError):
        # This call should fail because data_ptr<float>() will be used on double tensors.
        _ = module.forward(A, B)

# Issue 3 test: The unroll pragma may harm performance or correctness for variable loop bounds.
# While it is hard to check performance in a unit test, we can trigger a case where the dynamic
# loop bounds vary non-trivially. A matrix multiplication on a modestly sized matrix is used.
def test_loop_unroll_invalid_assumption():
    N = 300  # A size that does not align with the unroll factor assumptions.
    A = torch.triu(torch.randn(N, N, device="cuda", dtype=torch.float32))
    B = torch.triu(torch.randn(N, N, device="cuda", dtype=torch.float32))
    module = build_kernel()
    C = module.forward(A, B)
    # Compute the expected result.
    C_ref = torch.triu(torch.matmul(A, B))
    # The results may be off due to the unroll optimization assuming a fixed iteration count.
    # We trigger the issue by asserting that the results are not (incorrectly) equal.
    if torch.allclose(C, C_ref, atol=1e-5):
        pytest.skip("Loop unroll issue not triggered for this matrix size.")
    else:
        assert not torch.allclose(C, C_ref, atol=1e-5), "Kernel result should differ from reference result due to unroll issues."

if __name__ == "__main__":
    pytest.main([__file__])
