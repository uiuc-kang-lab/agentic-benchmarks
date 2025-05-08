
import pytest
import torch
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

# Issue 1: Passing a double tensor (non-float32) should cause an error due to incorrect pointer type.
def test_double_tensor_input():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    module = build_kernel()
    # Create a double tensor on CUDA
    x = torch.randn(1024, dtype=torch.double, device='cuda')
    with pytest.raises(RuntimeError):
        # This should raise an error, because the kernel expects float32 (casting issue)
        module.forward(x)
    torch.cuda.synchronize()

# Issue 2: Passing a CPU tensor should cause an error because the kernel expects a CUDA tensor.
def test_cpu_tensor_input():
    module = build_kernel()
    # Create a tensor on CPU
    x = torch.randn(1024, dtype=torch.float32, device='cpu')
    with pytest.raises(RuntimeError):
        # This should raise an error since the kernel is launched on CUDA.
        module.forward(x)

# Issue 3: Kernel launch errors might go undetected. We can try to force an error by giving an invalid size.
# Here we simulate a misconfiguration by passing an absurdly small grid configuration.
def test_kernel_launch_failure():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    module = build_kernel()
    # Create a valid CUDA tensor but we will monkey-patch the tensor's numel such that blocks are zero.
    # Since we can't directly force a kernel launch error from Python, we simulate the error by
    # passing a tensor with 0 elements.
    x = torch.empty(0, dtype=torch.float32, device='cuda')
    # In this case, the kernel doesn't do any work, but if error checking was added in the kernel,
    # launching with 0 elements might be caught. Here we check that the function returns an empty tensor.
    y = module.forward(x)
    assert y.numel() == 0

# Issue 4: Although not causing immediate runtime error, using input.type() instead of input.scalar_type()
# is deprecated. We can simulate this in the test by checking warnings.
def test_deprecated_dtype_usage_warning():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    module = build_kernel()
    x = torch.randn(1024, dtype=torch.float32, device='cuda')
    # We capture warnings to check if any deprecation warning is raised.
    with pytest.warns(UserWarning):
        y = module.forward(x)
    # Verify output correctness approximately.
    y_ref = torch.tanh(x)
    torch.cuda.synchronize()
    assert torch.allclose(y, y_ref, atol=1e-5)
