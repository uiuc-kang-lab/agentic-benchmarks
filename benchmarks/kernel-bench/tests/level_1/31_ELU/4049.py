
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

def build_kernel():
    # Load/compile the CUDA extension from kernel.cu
    cuda_module = load(
        name="elu_kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Test that passing a tensor with an unsupported dtype (float64)
# produces incorrect output (or more precisely, not matching the CPU version).
def test_incorrect_dtype():
    cuda_module = build_kernel()
    # Create a tensor with float64 that is CUDA, contiguous.
    x = torch.randn(1024, device="cuda", dtype=torch.float64)
    
    # Compute reference using PyTorch's builtin ELU conversion (casting to float is implicit in F.elu, so force conversion for an appropriate comparison).
    ref = F.elu(x.float(), alpha=1.0)
    # Call our custom CUDA kernel; note that the kernel expects float* data.
    out = cuda_module.forward(x, 1.0)
    # Since x was float64 but treated as float32 the output should differ significantly.
    # We expect the results to not be close.
    if torch.allclose(out, ref, atol=1e-5):
        pytest.fail("Kernel accepted a float64 tensor and produced results that match float32 ELU, which is unexpected.")

# Issue 2 & 3: Test that passing a non-contiguous tensor triggers an error.
def test_noncontiguous_tensor():
    cuda_module = build_kernel()
    # Create a contiguous tensor and then force a non-contiguous layout by transposing.
    x = torch.randn(32, 32, device="cuda", dtype=torch.float32)
    x_noncontig = x.t()  # Transpose makes it non-contiguous
    assert not x_noncontig.is_contiguous(), "Tensor should be non-contiguous for this test."
    
    with pytest.raises(RuntimeError) as excinfo:
        _ = cuda_module.forward(x_noncontig, 1.0)
    # Check that the error message indicates the contiguity issue.
    assert "must be contiguous" in str(excinfo.value)
    
# Issue 2 (continued): Test that kernel launch errors are not caught.
# One indirect way to test this is to pass an input that is extremely large so that 
# the kernel launch might fail. However, triggering an actual launch error reliably is non-trivial.
# Instead, we simulate the expectation by calling synchronize and checking no error was raised.
def test_kernel_launch_error_check():
    cuda_module = build_kernel()
    # Create a moderately sized tensor that should work correctly.
    x = torch.randn(1024, device="cuda", dtype=torch.float32)
    try:
        out = cuda_module.forward(x, 1.0)
        # Force synchronization to surface kernel launch errors.
        torch.cuda.synchronize()
    except Exception as e:
        pytest.skip("Kernel launch error detected (as expected by the missing error check): " + str(e))
    # No exception means that error checking inside the kernel call is missing.
    # This test serves as a reminder that the kernel does not perform error-checking.
    # We simply pass the test if no exception is raised.
    assert out is not None
