
import pytest
import torch
from torch.utils.cpp_extension import load

# Build/load the CUDA kernel extension from 'kernel.cu'
def build_kernel():
    return load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )

@pytest.fixture(scope="module")
def kernel_module():
    mod = build_kernel()
    return mod

# Test 1: Verify that using a non-float32 tensor (e.g. double) triggers an error.
def test_dtype_issue(kernel_module):
    # Create a tensor of type double on CUDA.
    x = torch.randn(16, 256, 256, device="cuda", dtype=torch.double)
    reduction_dim = 1
    with pytest.raises(RuntimeError) as exc_info:
        # The kernel expects a float pointer, so passing double should raise an error.
        out = kernel_module.forward(x, reduction_dim)
        # Synchronize to trigger potential CUDA launch errors.
        torch.cuda.synchronize()
    assert "must be a CUDA tensor" not in str(exc_info.value), "Unexpected error message."

# Test 2: Verify that providing a negative reduction dimension triggers an error.
def test_negative_dim(kernel_module):
    # Create a float32 tensor on CUDA.
    x = torch.randn(16, 256, 256, device="cuda", dtype=torch.float32)
    negative_dim = -1  # PyTorch accepts negative dims, but our kernel does not handle them.
    with pytest.raises(IndexError) as exc_info:
        out = kernel_module.forward(x, negative_dim)
        torch.cuda.synchronize()
    # Depending on how the permutation is handled, the error might be thrown either in C++ or later.
    # Here we assume an IndexError (or another error) is raised.
    assert "out of range" in str(exc_info.value) or "Invalid" in str(exc_info.value), \
        "Negative dimension did not trigger an expected error."

# Test 3: Verify that an out-of-range reduction dimension triggers an error.
def test_out_of_range_dim(kernel_module):
    # Create a float32 tensor on CUDA.
    x = torch.randn(16, 256, 256, device="cuda", dtype=torch.float32)
    invalid_dim = 5  # For a 3D tensor, valid dims are 0, 1, 2.
    with pytest.raises(IndexError) as exc_info:
        out = kernel_module.forward(x, invalid_dim)
        torch.cuda.synchronize()
    assert "out of range" in str(exc_info.value) or "Invalid" in str(exc_info.value), \
        "Out-of-range dimension did not trigger an expected error."
