
import math
import torch
import pytest
from torch.utils.cpp_extension import load
import threading

# Utility function to build and load the CUDA extension from kernel.cu
def build_kernel():
    return load(
        name="maxpool3d_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

# Issue 1: The use of std::numeric_limits<scalar_t>::infinity() in device code may fail
# for types where this is not supported on-device. One way to trigger a problem is to pass
# an input tensor of a type for which the device version of std::numeric_limits is not available.
def test_issue_infinity():
    kernel_module = build_kernel()
    # Create an input tensor of type double (which is sometimes not handled by such device code)
    # Note: The kernel dispatch is done with AT_DISPATCH_FLOATING_TYPES, so double is included,
    # but device-side support for infinity using std::numeric_limits<double>::infinity() may be problematic.
    batch_size, channels, d, h, w = 2, 2, 8, 8, 8
    x = torch.randn(batch_size, channels, d, h, w, device='cuda', dtype=torch.double)
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1
    # This call will invoke set_const_params with the parameters and then the kernel.
    # If std::numeric_limits<T>::infinity() is not properly supported in device code,
    # the kernel might generate wrong results.
    output = kernel_module.forward(x, kernel_size, stride, padding, dilation, False, False)
    # We perform a rough check: output values should not be -infinity.
    if torch.any(output == float("-inf")):
        pytest.fail("Output contains -infinity values, indicating an issue with std::numeric_limits usage on device")

# Issue 2: The kernel does not support non-uniform (per-dimension) parameters.
# We simulate the situation by mimicking a call with non-uniform pooling parameters.
# Since the C++ interface only takes int parameters, we try calling it with parameters that
# would be valid for a uniform operation but then compare against a PyTorch CPU version that can
# handle non-uniform parameters. In our test we “simulate” non-uniformity by altering the spatial
# dimensions of the input such that the uniform kernel parameters lead to a discrepancy.
def test_issue_nonuniform():
    kernel_module = build_kernel()
    # Create an input tensor where the spatial dimensions are different.
    batch_size, channels = 1, 1
    # Make non-cubic spatial dimensions.
    d, h, w = 10, 20, 30
    x = torch.randn(batch_size, channels, d, h, w, device='cuda', dtype=torch.float32)
    # We intend to use different pooling parameters per dim, but our kernel only accepts one value each.
    # For example, a real non-uniform pooling might use kernel_size=(3,4,5), etc.
    # Here we supply uniform parameters, but then compare against PyTorch's own MaxPool3d supporting tuples.
    # The kernel, however, is not equipped to work with tuple parameters.
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1
    # Run our custom kernel
    output_custom = kernel_module.forward(x, kernel_size, stride, padding, dilation, False, False)
    # Run PyTorch built-in max_pool3d with an equivalent uniform setting
    output_ref = torch.nn.functional.max_pool3d(x, kernel_size=kernel_size, stride=stride,
                                                  padding=padding, dilation=dilation, ceil_mode=False)
    # If the custom kernel were generalized to support per-dimension parameters, we might expect a matching output.
    # Here, since the custom kernel isn’t general, the discrepancy will be exposed when comparing to the reference.
    if not torch.allclose(output_custom, output_ref, atol=1e-4):
        pytest.fail("Custom kernel does not handle non-uniform input dimensions as expected.")

# Issue 3: The use of __constant__ memory for parameters may cause concurrency issues.
# We simulate this by running two kernel launches concurrently with different parameters.
def test_issue_constant_memory_concurrency():
    kernel_module = build_kernel()
    batch_size, channels = 1, 1
    d, h, w = 16, 16, 16
    # Create two different inputs and parameters.
    x1 = torch.randn(batch_size, channels, d, h, w, device='cuda', dtype=torch.float32)
    x2 = torch.randn(batch_size, channels, d, h, w, device='cuda', dtype=torch.float32)
    
    # Parameters for first kernel call.
    params1 = {'kernel_size': 2, 'stride': 2, 'padding': 0, 'dilation': 1, 'ceil_mode': False}
    # Parameters for second kernel call.
    params2 = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'ceil_mode': False}

    # Holder for outputs.
    outputs = {}

    def call_kernel(key, x_input, params):
        outputs[key] = kernel_module.forward(x_input,
                                             params['kernel_size'],
                                             params['stride'],
                                             params['padding'],
                                             params['dilation'],
                                             False,
                                             params['ceil_mode'])

    # Launch two threads that call the kernel concurrently.
    thread1 = threading.Thread(target=call_kernel, args=("out1", x1, params1))
    thread2 = threading.Thread(target=call_kernel, args=("out2", x2, params2))
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()

    # Run PyTorch's built-in max_pool3d to get reference outputs.
    ref_out1 = torch.nn.functional.max_pool3d(x1, kernel_size=params1['kernel_size'],
                                                stride=params1['stride'],
                                                padding=params1['padding'],
                                                dilation=params1['dilation'],
                                                ceil_mode=params1['ceil_mode'])
    ref_out2 = torch.nn.functional.max_pool3d(x2, kernel_size=params2['kernel_size'],
                                                stride=params2['stride'],
                                                padding=params2['padding'],
                                                dilation=params2['dilation'],
                                                ceil_mode=params2['ceil_mode'])
    # Compare both outputs with their references.
    if not torch.allclose(outputs["out1"], ref_out1, atol=1e-4):
        pytest.fail("Concurrent kernel call with params1 produced unexpected results, indicating __constant__ memory pollution.")
    if not torch.allclose(outputs["out2"], ref_out2, atol=1e-4):
        pytest.fail("Concurrent kernel call with params2 produced unexpected results, indicating __constant__ memory pollution.")
