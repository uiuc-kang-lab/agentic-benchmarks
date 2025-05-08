
import torch
import pytest
from torch.utils.cpp_extension import load

# Helper function to build the CUDA extension.
def build_kernel():
    cuda_module = load(
        name="max_pool2d_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Incorrect thread block mapping.
# Use parameters such that output dimensions are not multiples of 32.
def test_thread_block_mapping_issue():
    # Create an input tensor whose output dimensions will be non-multiple of 32.
    # For example, with kernel_size=3, stride=2, padding=1, dilation=1, for an input of 23x23,
    # the output will be floor((23+2-3)/2)+1 = floor(22/2)+1 = 11+1 = 12 (non-multiple of 32).
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1

    # Create a relatively small input tensor.
    input_tensor = torch.randn(1, 1, 23, 23, device='cuda', dtype=torch.float32)
    
    # PyTorch reference:
    ref = torch.nn.functional.max_pool2d(
        input_tensor, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation
    )
    
    cuda_module = build_kernel()
    out = cuda_module.forward(input_tensor, kernel_size, stride, padding, dilation)
    torch.cuda.synchronize()

    # This test is expected to fail (or report discrepancies) if the mapping issue occurs.
    assert torch.allclose(out, ref, atol=1e-5), f"Thread block mapping issue: max pool result does not match reference. Difference: {(out - ref).abs().max()}"

# Issue 2: Use of std::numeric_limits<scalar_t>::infinity() may fail for half precision.
def test_numeric_limits_issue_half_precision():
    kernel_size = 2
    stride = 2
    padding = 0
    dilation = 1

    # Create a half precision input, which might trigger problems with std::numeric_limits<scalar_t>::infinity()
    input_tensor = torch.randn(1, 1, 32, 32, device='cuda', dtype=torch.float16)
    
    # Use PyTorch’s functional max_pool2d for reference.
    ref = torch.nn.functional.max_pool2d(
        input_tensor.float(), kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation
    ).half()
    
    cuda_module = build_kernel()
    out = cuda_module.forward(input_tensor, kernel_size, stride, padding, dilation)
    torch.cuda.synchronize()

    assert torch.allclose(out, ref, atol=1e-5), f"Numeric limits issue: half precision max pool result does not match reference. Diff: {(out - ref).abs().max()}"

# Issue 3: Unqualified call to max() may lead to compilation/logic errors when operating on double precision.
def test_unqualified_max_issue_double_precision():
    kernel_size = 2
    stride = 2
    padding = 0
    dilation = 1

    # Create a double precision input.
    input_tensor = torch.randn(1, 1, 32, 32, device='cuda', dtype=torch.float64)
    
    ref = torch.nn.functional.max_pool2d(
        input_tensor, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation
    )
    
    cuda_module = build_kernel()
    out = cuda_module.forward(input_tensor, kernel_size, stride, padding, dilation)
    torch.cuda.synchronize()
    
    assert torch.allclose(out, ref, atol=1e-5), f"Unqualified max() issue: double precision result mismatch. Diff: {(out - ref).abs().max()}"

# Issue 4: Deprecated API for type dispatch.
# Here we simply trigger the kernel with an unsupported type configuration.
def test_deprecated_api_dispatch_issue():
    kernel_size = 2
    stride = 2
    padding = 0
    dilation = 1

    # Although the kernel is written using AT_DISPATCH_FLOATING_TYPES(input.type(), ...),
    # we trigger it with a tensor of a type that might be used in modern codes (e.g., torch.float32).
    # This test will pass if the output is correct but highlights the need to update the dispatch macro.
    input_tensor = torch.randn(1, 1, 64, 64, device='cuda', dtype=torch.float32)
    
    ref = torch.nn.functional.max_pool2d(
        input_tensor, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation
    )
    
    cuda_module = build_kernel()
    out = cuda_module.forward(input_tensor, kernel_size, stride, padding, dilation)
    torch.cuda.synchronize()
    
    assert torch.allclose(out, ref, atol=1e-5), f"Deprecated API dispatch issue: output mismatch. Diff: {(out - ref).abs().max()}"

# Issue 5: Overly rigid launch configuration.
# With a very small output tensor the fixed 1024 threads per block may trigger spurious computations.
def test_launch_configuration_issue():
    # Choose parameters that yield a very small output (e.g., 1x1 output)
    kernel_size = 5
    stride = 1
    padding = 2
    dilation = 1

    input_tensor = torch.randn(1, 1, 3, 3, device='cuda', dtype=torch.float32)
    # In this case, the output of a 3x3 input with kernel size 5 and padding 2 is still 3x3,
    # but we can try an even smaller input to see if extra threads (from fixed 1024 threads per block)
    # cause trouble.
    ref = torch.nn.functional.max_pool2d(
        input_tensor, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation
    )
    
    cuda_module = build_kernel()
    out = cuda_module.forward(input_tensor, kernel_size, stride, padding, dilation)
    torch.cuda.synchronize()
    
    assert torch.allclose(out, ref, atol=1e-5), f"Launch configuration issue: computed output does not match reference. Diff: {(out - ref).abs().max()}"
