
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build the CUDA extension module from kernel.cu
def build_kernel():
    module = load(
        name="hardsigmoid_kernel",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return module

# Issue 1: Lack of device synchronization might hide errors.
# We simulate an error scenario by intentionally passing an incorrect tensor (e.g. CPU tensor)
# which should trigger the TORCH_CHECK(input.is_cuda(), ...) in forward.
def test_cpu_input_raises():
    kernel_module = build_kernel()
    x = torch.randn(16, 16384)  # CPU tensor
    with pytest.raises(RuntimeError, match="Input tensor must be on CUDA"):
        _ = kernel_module.forward(x)

# Issue 2: The kernel does not support half precision.
def test_half_precision_not_supported():
    kernel_module = build_kernel()
    # Create a half precision tensor on CUDA
    x = torch.randn(16, 16384, device='cuda', dtype=torch.float16)
    with pytest.raises(RuntimeError, match="AT_DISPATCH_FLOATING_TYPES"):
        _ = kernel_module.forward(x)
        
# Issue 3: The kernel assumes contiguous tensors.
# We provide a non-contiguous tensor (by transposing a contiguous tensor) to see if the result differs 
# from the expected hardsigmoid output computed by PyTorch.
def test_non_contiguous_input():
    kernel_module = build_kernel()
    # Create a contiguous tensor and then make it non-contiguous via transpose
    a = torch.randn(64, 64, device='cuda', dtype=torch.float32)
    # Transpose makes it non-contiguous if size > 1 in both dims
    x = a.t()
    # Compute reference result using PyTorch's hardsigmoid on the non-contiguous tensor,
    # forcing a contiguous result.
    ref = torch.nn.functional.hardsigmoid(x)
    # Call the CUDA kernel extension (which uses data_ptr and expects contiguous layout)
    out = kernel_module.forward(x)
    # Since the kernel does not adjust for non-contiguous strides, it is likely to produce a different result.
    # We check that the outputs are not almost equal.
    assert not torch.allclose(out, ref, atol=1e-5), \
        "Kernel seems to work with non-contiguous tensors, but it should assume contiguous data."
        
if __name__ == '__main__':
    pytest.main([__file__])
