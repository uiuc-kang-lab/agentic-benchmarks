
import torch
import pytest
from torch.utils.cpp_extension import load
import math

def build_kernel():
    cuda_module = load(
        name="l1_norm_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Helper function to perform L1 normalization in Python
def l1_normalize_cpu(x: torch.Tensor) -> torch.Tensor:
    norm = torch.sum(torch.abs(x), dim=1, keepdim=True)
    # Avoid division by zero
    norm = norm.clamp(min=1e-12)
    return x / norm

# Issue 1: Tensor dtypes other than float32 are not checked.
def test_wrong_dtype():
    cuda_module = build_kernel()
    batch_size = 16
    dim = 16384
    # Create a double tensor (float64) which should not be supported.
    x = torch.randn(batch_size, dim, dtype=torch.float64, device='cuda')
    # Since the kernel uses x.data_ptr<float>(), the result will be invalid.
    with pytest.raises(Exception):
        # Expect that the wrong type eventually leads to an error or misnormalized result.
        out = cuda_module.forward(x)
        torch.cuda.synchronize()

# Issue 2: Misaligned memory access due to assumptions of 16-byte alignment.
def test_misaligned_input():
    cuda_module = build_kernel()
    batch_size = 16
    dim = 16384
    # Create a tensor with an extra column, then slice to force a potential misalignment.
    x_full = torch.randn(batch_size, dim + 1, dtype=torch.float32, device='cuda')
    # Slicing along the second dimension may yield a tensor whose data_ptr is not 16-byte aligned.
    x = x_full.narrow(1, 1, dim)
    # Compute expected normalization on CPU
    expected = l1_normalize_cpu(x).cpu()
    out = cuda_module.forward(x)
    torch.cuda.synchronize()
    # Because of misalignment, the float4 loads might result in undefined behavior.
    # We check that the CUDA result does NOT match the expected normalization.
    # (If the kernel were correct, these would match. Here we are testing that the misalignment issue is triggered.)
    if torch.allclose(out.cpu(), expected, atol=1e-5):
        pytest.fail("Kernel did not trigger misaligned memory access issue as expected.")

# Issue 3: Unnecessary use of dynamic shared memory vs. static declaration.
# This issue might not trigger a runtime failure but can be highlighted by checking that additional dynamic shared memory has no effect.
# Here we simulate a case where a more complex kernel might use dynamic shared memory.
def test_dynamic_shared_memory_usage():
    cuda_module = build_kernel()
    batch_size = 16
    dim = 16384
    # Create a standard float32 input.
    x = torch.randn(batch_size, dim, dtype=torch.float32, device='cuda')
    expected = l1_normalize_cpu(x)
    out = cuda_module.forward(x)
    torch.cuda.synchronize()
    # Even if the kernel produces normalized outputs, the use of dynamic shared memory is inconsistent.
    # We compare the result with the expected to see if any unintended artifacts appear.
    if not torch.allclose(out, expected, atol=1e-5):
        pytest.fail("Kernel output with dynamic shared memory parameter does not match expected normalization.")

if __name__ == "__main__":
    pytest.main([__file__])
