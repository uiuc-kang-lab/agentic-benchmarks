
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="leaky_relu_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_duplicate_processing():
    # Create an input size large enough so that each thread performs multiple iterations.
    # This forces duplicate processing because the loop unrolling oversteps iterations.
    n = 1280  # Using a size that results in gridDim > 1; this will cause overlapping work.
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    # Use the same negative_slope as in the kernel launch.
    negative_slope = 0.01
    ref = torch.nn.functional.leaky_relu(x, negative_slope=negative_slope)
    
    cuda_module = build_kernel()
    out = cuda_module.forward(x, negative_slope)
    torch.cuda.synchronize()
    
    # Because of duplicate processing the kernel output is expected to be incorrect (different from the correct activation)
    # The test will pass if the kernel output does not match the correct output.
    assert not torch.allclose(out, ref, atol=1e-5), "The kernel output unexpectedly matches the reference; duplicate processing issue not triggered."

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_type_mismatch():
    # Create an input tensor with type double. Since the kernel always interprets the data as float,
    # the output will be computed wrongly.
    n = 1024
    x = torch.randn(n, device="cuda", dtype=torch.double)
    negative_slope = 0.01
    ref = torch.nn.functional.leaky_relu(x, negative_slope=negative_slope)
    
    cuda_module = build_kernel()
    out = cuda_module.forward(x, negative_slope)
    torch.cuda.synchronize()
    
    # The output is expected to be incorrect because the kernel misinterprets data type.
    assert not torch.allclose(out, ref, atol=1e-5), "The kernel output unexpectedly matches the reference; type mismatch issue not triggered."

# (Optional) A test for kernel launch error checking might be added when the kernel is extended to more complex cases.
