
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

# Issue 1: Unnecessary atomic operation.
# Although correctness is maintained, we design a test to check for correct output and rely on performance hints.
def test_atomic_exch_usage_correctness():
    # Create a large input tensor to emphasize potential performance impacts.
    N = 1 << 20  # 1M elements
    x = torch.randn(N, device="cuda", dtype=torch.float32)
    module = build_kernel()
    out = module.forward(x)
    expected = x / (1.0 + torch.abs(x))
    # Even if the atomic is used, the result should be correct.
    assert torch.allclose(out, expected, atol=1e-5), "Output incorrect - potential atomic operation misuse."

# Issue 2: Redundant use of shared memory.
# The test verifies that the use of shared memory (which might cause issues in more complex kernels)
# does not affect the correctness of the computation.
def test_shared_memory_usage_correctness():
    N = 1 << 18  # moderate-sized tensor
    x = torch.randn(N, device="cuda", dtype=torch.float32)
    module = build_kernel()
    out = module.forward(x)
    expected = x / (1.0 + torch.abs(x))
    assert torch.allclose(out, expected, atol=1e-5), "Output incorrect - potential shared memory misuse."

# Issue 3: Lack of support for data types other than float32.
# This test passes a float64 tensor; the kernel should reject it.
def test_dtype_restriction():
    x = torch.randn(1024, device="cuda", dtype=torch.float64)
    module = build_kernel()
    with pytest.raises(RuntimeError):
        # Expect runtime error due to type mismatch since kernel explicitly assumes float.
        module.forward(x)
