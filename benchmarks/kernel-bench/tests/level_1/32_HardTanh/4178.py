
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# Helper function to compile and load the CUDA extension
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# 1. Test case for double input type (issue #1)
# The kernel uses fmaxf/fminf which are for float, so when we pass a double tensor,
# we expect the behavior to be incorrect or to raise an exception.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_double_dtype():
    my_module = build_kernel()
    # Create a tensor of double type, which is not properly handled by fmaxf/fminf.
    x = torch.randn(1024, device="cuda", dtype=torch.float64)
    # We expect the CUDA kernel to either produce a wrong result or throw an error.
    # Here, we catch a RuntimeError if thrown.
    with pytest.raises(RuntimeError):
        # The forward function expects a CUDA tensor; type double might trigger an error.
        my_module.forward(x, -1.0, 1.0)
        
# 2. Test case for non-contiguous tensor input (issue #2)
# Create a non-contiguous tensor by transposing a 2D tensor.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_non_contiguous():
    my_module = build_kernel()
    # Create a contiguous tensor and force a non-contiguous view by transposing it.
    x = torch.randn(32, 33, device="cuda", dtype=torch.float32)
    x_noncontig = x.transpose(0, 1)  # this tensor is non-contiguous
    # Compute expected result via PyTorch's built-in function.
    expected = F.hardtanh(x_noncontig.clone(), min_val=-1.0, max_val=1.0)
    # Run the CUDA kernel (which does not account for non-contiguous strides).
    result = my_module.forward(x_noncontig, -1.0, 1.0)
    torch.cuda.synchronize()
    # Because the kernel ignores strides, the output will differ from the expected outcome.
    assert not torch.allclose(result, expected, atol=1e-5), "Kernel produced correct results on a non-contiguous tensor even though it is expected to fail."

# 3. Test case for CPU tensor input to trigger the explicit check in the forward function.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_input():
    my_module = build_kernel()
    # Create a CPU tensor (even though our module expects a CUDA tensor)
    x_cpu = torch.randn(1024, dtype=torch.float32)
    with pytest.raises(ValueError):
        # The forward() function is supposed to throw std::invalid_argument,
        # which translates to a RuntimeError in Python. We check for an exception.
        my_module.forward(x_cpu, -1.0, 1.0)
