
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the extension from kernel.cu.
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Misaligned memory access
# We create a tensor whose data pointer is likely misaligned for float4 loads.
# We do this by allocating extra elements and slicing off the first element.
@pytest.mark.xfail(reason="Kernel does not check for proper 16-byte alignment – misaligned memory access may occur.")
def test_misaligned_tensor():
    M = 128
    N = 128
    total = M * N + 1  # extra element to allow misalignment
    base = torch.randn(total, device="cuda", dtype=torch.float32)
    # Create a misaligned tensor by ignoring the first element.
    A = base.narrow(0, 1, M * N).reshape(M, N)
    s = 3.14
    kernel = build_kernel()
    # This call may result in incorrect output or even a crash on some systems.
    C = kernel.forward(A, s)
    torch.cuda.synchronize()
    expected = A * s
    # We expect the kernel output to be incorrect when memory is misaligned.
    assert not torch.allclose(C, expected, atol=1e-5), "Kernel unexpectedly produced correct results for misaligned tensor."

# Issue 2: Noncontiguous input tensor
# If the input tensor is noncontiguous (for example, by transposing), then the reinterpretation as float4 is invalid.
@pytest.mark.xfail(reason="Kernel assumes contiguous input, so noncontiguous tensors may produce incorrect results.")
def test_noncontiguous_tensor():
    M = 128
    N = 128
    A = torch.randn(M, N, device="cuda", dtype=torch.float32).t()  # transposed -> noncontiguous
    s = 3.14
    kernel = build_kernel()
    C = kernel.forward(A, s)
    torch.cuda.synchronize()
    expected = A * s
    assert not torch.allclose(C, expected, atol=1e-5), "Kernel unexpectedly produced correct results for a noncontiguous tensor."

# Issue 3: Block configuration dependency on warp size
# Although the kernel hard-codes 256 threads per block (which is a multiple of 32),
# a future change or misuse might trigger the issue.
# We simulate a condition by creating a tensor whose number of elements forces an uneven grid mapping.
@pytest.mark.xfail(reason="Kernel assumes blockDim.x is a multiple of 32; uneven grid mappings are not handled properly.")
def test_uneven_grid_mapping():
    # Choose numbers that force the float4Count to be very small
    # so the calculated grid may process out-of-bound indices.
    M = 31  # not a multiple of warp size when divided by 4
    N = 31
    A = torch.randn(M, N, device="cuda", dtype=torch.float32)
    s = 3.14
    kernel = build_kernel()
    C = kernel.forward(A, s)
    torch.cuda.synchronize()
    expected = A * s
    assert not torch.allclose(C, expected, atol=1e-5), "Kernel unexpectedly produced correct results for a grid mapping issue."

# Issue 4: Unnecessary __syncwarp() call 
# Although the __syncwarp() call itself might not change the computation,
# it is unnecessary and in more complex scenarios with divergent execution, 
# it may adversely affect performance.
# This issue is hard to trigger via a correctness test,
# but we include a dummy test to highlight its presence.
@pytest.mark.xfail(reason="Kernel uses an unnecessary __syncwarp() call that may degrade performance.")
def test_unnecessary_syncwarp():
    M = 1024
    N = 1024
    A = torch.randn(M, N, device="cuda", dtype=torch.float32)
    s = 3.14
    kernel = build_kernel()
    # Run several iterations to accumulate potential performance degradation,
    # though correctness might still hold.
    C = kernel.forward(A, s)
    torch.cuda.synchronize()
    expected = A * s
    # We purposely expect the correct result to be returned here because syncwarp doesn't affect correctness.
    # So we mark the test as xfail due to performance concerns rather than numerical incorrectness.
    assert torch.allclose(C, expected, atol=1e-5), "Kernel failed to compute correct results despite __syncwarp() usage."
