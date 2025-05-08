
import pytest
import torch
from torch.utils.cpp_extension import load
import os

def build_kernel():
    # Force recompile by using a unique build directory
    module = load(
        name="optimized_softplus",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

# Issue 1: Test for lack of half precision support.
def test_half_precision_not_supported():
    # Create a half-precision tensor on CUDA
    x = torch.randn(1024, device="cuda", dtype=torch.float16)
    my_kernel = build_kernel()
    with pytest.raises(RuntimeError):
        # The extension as written dispatches only float and double,
        # so using half should result in an error.
        y = my_kernel.forward(x)
        torch.cuda.synchronize()

# Issue 2: Test for non-contiguous tensor input.
def test_non_contiguous_input():
    # Create a contiguous tensor and then make it non-contiguous by transposing.
    x = torch.randn(64, 256, device="cuda", dtype=torch.float32)
    x_t = x.t()  # This tensor is non-contiguous.
    my_kernel = build_kernel()
    # We expect the kernel to produce wrong results if it assumes contiguity.
    # Compare with torch.nn.functional.softplus.
    y_kernel = my_kernel.forward(x_t)
    torch.cuda.synchronize()
    y_reference = torch.nn.functional.softplus(x_t)
    # The outputs may not match if non-contiguity is not handled correctly.
    assert not torch.allclose(y_kernel, y_reference, atol=1e-5), \
        "Kernel unexpectedly produced correct output on non-contiguous tensor."

# Issue 3: Test for missing CUDA error checking.
def test_launch_error_detection():
    # Create an input tensor on CPU instead of CUDA.
    # The kernel expects a CUDA tensor; passing a CPU tensor should trigger an error.
    x_cpu = torch.randn(1024, dtype=torch.float32)
    my_kernel = build_kernel()
    with pytest.raises(RuntimeError):
        # This should raise an error because the kernel is launched on GPU.
        y = my_kernel.forward(x_cpu)
        torch.cuda.synchronize()
        
if __name__ == "__main__":
    pytest.main([__file__])
