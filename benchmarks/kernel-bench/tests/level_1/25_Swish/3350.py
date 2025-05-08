
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    kernel_module = load(
        name="swish_kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return kernel_module

def swish_reference(x: torch.Tensor) -> torch.Tensor:
    # Compute swish in torch with proper dtype
    return x * torch.sigmoid(x)

def test_wrong_dtype():
    # Issue 1: Passing a tensor with a non-float32 type.
    # The kernel assumes float32, so providing a double tensor should yield an incorrect result.
    module = build_kernel()
    N = 1024
    # Create a double tensor on CUDA.
    x = torch.randn(N, device="cuda", dtype=torch.double)
    # Convert to double swish reference
    ref = swish_reference(x)
    # Call the kernel extension (kernel will treat the data as float)
    y = module.forward(x)
    torch.cuda.synchronize()
    # Because x is double but kernel computes with float, result will not be equal to ref.
    # We assert that the output is NOT close to the correct result.
    if torch.allclose(y.to(torch.double), ref, atol=1e-5):
        pytest.fail("Kernel incorrectly handled non-float32 tensor without errors.")
    else:
        # Test passes if output is different from the expected double-precision result.
        assert True

def test_cpu_tensor_error():
    # Issue 1 (and partly issue 2): The kernel requires a CUDA tensor.
    module = build_kernel()
    x_cpu = torch.randn(1024, dtype=torch.float32, device="cpu")
    with pytest.raises(RuntimeError, match="Input tensor must be on CUDA"):
        _ = module.forward(x_cpu)

def test_no_cuda_error_checking():
    # Issue 2: The kernel launch does not check for errors.
    # One way to potentially trigger an error is to use an extremely large tensor that could launch an invalid kernel configuration.
    # While this test may not always trigger a kernel error, we try to provoke one by allocating a huge tensor.
    module = build_kernel()
    try:
        # Try to allocate a huge tensor (the size may be adjusted to your device limits)
        huge_n = 2**31 // 4  # roughly 0.5 billion elements, adjust if needed to trigger error
        x = torch.empty(huge_n, dtype=torch.float32, device="cuda")
        y = module.forward(x)
        torch.cuda.synchronize()
        # If no error is raised, we warn about the lack of error checking.
        # In a correct kernel implementation, any launch error should be caught. Here we chose to mark this as a failure.
        pytest.fail("Kernel launch error was not detected despite using an extremely large tensor.")
    except RuntimeError:
        # If a runtime error is raised, that is acceptable.
        assert True
