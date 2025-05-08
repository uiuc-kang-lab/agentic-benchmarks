
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

# Issue 1: Kernel only supports float32. Test using float64 (double) inputs.
def test_input_tensor_dtype():
    my_module = build_kernel()
    M, K, N = 128, 16, 128
    # Use float64: kernel expects float32, so the results will be incorrect.
    A = torch.randn(M, K, dtype=torch.float64, device='cuda')
    B = torch.randn(K, N, dtype=torch.float64, device='cuda')
    with pytest.raises(RuntimeError):
        # This should fail because the kernel uses data_ptr<float>()
        C = my_module.forward(A, B)
        torch.cuda.synchronize()

# Issue 2: Kernel does not support batched inputs. Supply 3D tensors.
def test_batched_input():
    my_module = build_kernel()
    batch = 4
    M, K, N = 128, 16, 128
    # Create batched inputs (which are 3D) but our kernel only processes 2D tensors.
    A = torch.randn(batch, M, K, dtype=torch.float32, device='cuda')
    B = torch.randn(batch, K, N, dtype=torch.float32, device='cuda')
    # Attempting to use batched inputs should lead to incorrect behavior.
    # We simulate this by manually flattening only the first matrix in the batch to mimic wrong shape.
    with pytest.raises(IndexError):
        # This is expected to throw an error or produce an incorrect result
        # because the kernel's dimensions are derived from A.size(0) and A.size(1),
        # not processing the batch dimension.
        C = my_module.forward(A, B)
        # Here we force an index error or an assertion by comparing with the batched torch.matmul.
        C_ref = torch.matmul(A, B)
        torch.cuda.synchronize()
        # If kernel returns a result with a wrong shape, trying to index it as batched
        # should raise an error.
        _ = C[1]  # try to access a subsequent batch element

# Issue 3: Kernel does not check that A.size(1) == B.size(0).
def test_dimension_mismatch():
    my_module = build_kernel()
    M, K, N = 128, 16, 128
    # Intentionally create a dimension mismatch: B's first dimension is not equal to A's second.
    A = torch.randn(M, K, dtype=torch.float32, device='cuda')
    B = torch.randn(K + 1, N, dtype=torch.float32, device='cuda')
    with pytest.raises(RuntimeError):
        # Expecting a runtime or CUDA error due to out-of-bound memory access.
        C = my_module.forward(A, B)
        torch.cuda.synchronize()

# Issue 4: Kernel launch errors are not checked.
def test_kernel_launch_error():
    my_module = build_kernel()
    M, K, N = 128, 16, 128
    # Create valid inputs, but then deliberately pass an incorrect value for K to simulate a launch error.
    A = torch.randn(M, K, dtype=torch.float32, device='cuda')
    B = torch.randn(K, N, dtype=torch.float32, device='cuda')
    # Monkey-patch the forward function to pass an incorrect K value.
    # This simulates a scenario where the kernel is launched with wrong parameters.
    try:
        # Save the original forward function.
        orig_forward = my_module.forward
        # Wrap the forward to call with a wrong value of K.
        def wrong_forward(A, B):
            actual_K = A.size(1)
            wrong_K = actual_K + 5  # deliberately wrong
            M_val = A.size(0)
            N_val = B.size(1)
            # Create result tensor.
            C = torch.zeros((M_val, N_val), device=A.device, dtype=A.dtype)
            # Call the kernel with wrong_K. This is essentially a hack to simulate the issue.
            # Directly invoke the CUDA kernel with a wrong K.
            my_module.MatrixMulKernel(
                A.data_ptr(), B.data_ptr(), C.data_ptr(),
                M_val, N_val, wrong_K
            )
            torch.cuda.synchronize()
            return C
        my_module.forward = wrong_forward
        with pytest.raises(RuntimeError):
            C = my_module.forward(A, B)
            torch.cuda.synchronize()
    finally:
        # Restore the original forward.
        my_module.forward = orig_forward

# Issue 5: The unroll pragma might lead to problems if K is not a multiple of 4.
def test_unroll_mismatch():
    my_module = build_kernel()
    # Use a K value that is not divisible by 4.
    M, K, N = 128, 18, 128
    A = torch.randn(M, K, dtype=torch.float32, device='cuda')
    B = torch.randn(K, N, dtype=torch.float32, device='cuda')
    # Although the kernel loop is unrolled with 4 iterations,
    # it might compute incorrect results if the unrolling assumption is violated.
    C = my_module.forward(A, B)
    torch.cuda.synchronize()
    C_ref = torch.matmul(A, B)
    # We expect a mismatch here
    assert not torch.allclose(C, C_ref, atol=1e-5), "Kernel appears to handle non-multiple-of-4 K unexpectedly."

if __name__ == '__main__':
    pytest.main([__file__])
