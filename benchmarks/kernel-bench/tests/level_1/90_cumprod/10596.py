
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the CUDA extension module from kernel.cu
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

def test_non_contiguous_input():
    # Issue 1 & 2: Create a noncontiguous tensor by transposing a contiguous tensor.
    # For a tensor with shape (32, 64), transposing it makes the target dimension noncontiguous.
    a = torch.randn(32, 64, device="cuda", dtype=torch.float32)
    a_noncontig = a.t()  # Transpose produces a non-contiguous layout.
    # Here, we choose to perform the cumulative product along dimension 0 (first dimension).
    # The kernel using strides may compute wrong indices on this noncontiguous tensor.
    dim = 0
    cuda_module = build_kernel()
    # Invoke our CUDA kernel
    out_kernel = cuda_module.forward(a_noncontig, dim)
    # Use PyTorch native implementation as reference.
    out_reference = torch.cumprod(a_noncontig, dim=dim)
    # The outputs are expected to differ because the kernel indexing is based on contiguous assumptions.
    # We deliberately check that they are not close.
    assert not torch.allclose(out_kernel, out_reference, atol=1e-5), \
        "Kernel output unexpectedly matches torch.cumprod on a noncontiguous tensor."

def test_integer_input():
    # Issue 3: Using an integer tensor should trigger a runtime error because the kernel only supports
    # floating point and half types via the AT_DISPATCH_FLOATING_TYPES_AND_HALF macro.
    a_int = torch.randint(1, 10, (16, 32), device="cuda", dtype=torch.int32)
    dim = 1
    cuda_module = build_kernel()
    with pytest.raises(RuntimeError):
        # The forward call should raise an error due to unsupported data type.
        cuda_module.forward(a_int, dim)

# Optionally, one could add a test that runs with a contiguous tensor to verify the kernel works in the ideal case.
# However, since our focus is to trigger the issues, we omit that test here.
