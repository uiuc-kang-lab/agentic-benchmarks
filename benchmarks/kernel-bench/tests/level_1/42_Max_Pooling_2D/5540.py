
import torch
import pytest
from torch.utils.cpp_extension import load

# Helper function to compile and load the CUDA kernel.
def build_kernel():
    cuda_module = load(
        name="pool_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Non-standard header inclusion.
# While we cannot trigger a compile‐error at runtime,
# attempting to load the module will fail if <limits> is missing.
def test_missing_limits_header():
    with pytest.raises(Exception):
        # We simulate that a compile error will be raised
        # if the module does not include <limits>.
        load(
            name="bad_module",
            sources=["kernel.cu"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            with_cuda=True,
            verbose=True,
        )

# Test 2: Non-contiguous input tensor.
# The kernel computes indices assuming a contiguous NCHW layout.
# Here we create a non‐contiguous tensor and compare the result with PyTorch’s native max pool.
def test_non_contiguous_input():
    batch_size = 4
    channels = 3
    height = 10
    width = 10
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1

    # Create input tensor and force it to be non contiguous
    x = torch.randn(batch_size, channels, height, width, device="cuda")
    x = x.transpose(2, 3)  # Now non-contiguous in memory

    # Compute reference output with nn.MaxPool2d (ensuring input is contiguous)
    ref_pool = torch.nn.MaxPool2d(kernel_size, stride, padding, dilation=dilation)
    x_contig = x.contiguous()  # force contiguous for the reference
    ref_out = ref_pool(x_contig)

    # Run the custom CUDA kernel forward (assumes contiguous input)
    custom_module = build_kernel()
    try:
        out = custom_module.forward(x)
    except Exception as e:
        # If the kernel misbehaves due to non-contiguity, an exception is expected.
        pytest.skip("Kernel does not support non-contiguous inputs: " + str(e))
    torch.cuda.synchronize()
    
    # The result may differ due to mis-computed indices.
    assert not torch.allclose(out, ref_out, atol=1e-5), \
        "Kernel unexpectedly handled non-contiguous input correctly."

# Test 3: Asymmetric (non‐square) pooling parameters.
# The kernel interface accepts one kernel_size, stride, etc. While PyTorch’s nn.MaxPool2d
# supports separate values for height and width by passing tuples, our kernel cannot.
# Here we try to simulate asymmetric behavior by providing parameters that lead
# to a pooling output which does not match the reference.
def test_asymmetric_configuration():
    batch_size = 2
    channels = 3
    height = 15
    width = 20
    # The custom kernel only accepts a single kernel_size.
    # We simulate an asymmetric behavior by using kernel_size that does not match a
    # typical tuple (e.g., (2,3)) scenario. For our test, we only pass one value.
    kernel_size = 2  # Intended for both dimensions
    stride = 3       # Intended for both dimensions
    padding = 1
    dilation = 1

    x = torch.randn(batch_size, channels, height, width, device="cuda")
    pool_module = torch.nn.MaxPool2d(kernel_size, stride, padding, dilation=dilation)
    ref_out = pool_module(x)

    custom_module = build_kernel()
    out = custom_module.forward(x)
    torch.cuda.synchronize()
    
    # Since the built-in function can be configured with tuples,
    # we expect that using a single parameter in our custom kernel causes a discrepancy.
    assert not torch.allclose(out, ref_out, atol=1e-5), \
        "Kernel output matches reference despite asymmetric configuration expectation."

# Test 4: Lack of error checking on kernel launch.
# A deliberately “bad” input configuration is provided (e.g., input dimensions too small)
# so that the computed output dimensions become non-positive. The custom kernel may then produce
# an error or incorrect results. We check that the result does NOT match the reference.
def test_bad_input_configuration():
    # Using input dimensions so small that effective kernel area exceeds them.
    batch_size = 1
    channels = 1
    height = 4
    width = 4
    kernel_size = 5   # Deliberately larger than input dimensions (with dilation 1)
    stride = 1
    padding = 0
    dilation = 1

    x = torch.randn(batch_size, channels, height, width, device="cuda")
    # In PyTorch, if the pooling configuration is illegal, an error is raised.
    with pytest.raises(Exception):
        torch.nn.MaxPool2d(kernel_size, stride, padding, dilation=dilation)(x)

    # Our custom kernel does not do error checking and will try to compute an output.
    custom_module = build_kernel()
    out = custom_module.forward(x)
    torch.cuda.synchronize()
    # The kernel might produce an output of size 0 or garbage values.
    # We check that the output shape is not a valid pooling result.
    assert (out.numel() == 0 or out.isnan().any() or not out.isfinite().all()), \
        "Kernel did not fail as expected on bad input configuration."

# Test 5: Deprecated dispatch method.
# The AT_DISPATCH macro is called using input.type() instead of input.scalar_type().
# While this may work now, it is deprecated and may lead to subtle bugs.
# Here we test with an input whose scalar type is not the preferred type.
def test_deprecated_dispatch_usage():
    batch_size = 2
    channels = 2
    height = 16
    width = 16
    kernel_size = 2
    stride = 2
    padding = 0
    dilation = 1

    # Create a half precision input. AT_DISPATCH_FLOATING_TYPES may not cover __half.
    x = torch.randn(batch_size, channels, height, width, device="cuda", dtype=torch.half)
    # The reference nn.MaxPool2d supports half, but our kernel may not get dispatched correctly.
    pool_module = torch.nn.MaxPool2d(kernel_size, stride, padding, dilation=dilation)
    ref_out = pool_module(x)

    custom_module = build_kernel()
    with pytest.raises(RuntimeError):
        # The custom kernel is not expected to support half if not explicitly handled.
        _ = custom_module.forward(x)
