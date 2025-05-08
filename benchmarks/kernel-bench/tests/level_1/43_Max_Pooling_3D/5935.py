
import math
import torch
import pytest
from torch.utils.cpp_extension import load

# Load the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="max_pool3d_ext",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

@pytest.fixture(scope="module")
def cuda_kernel():
    module = build_kernel()
    return module

# A helper function to compute the expected PyTorch output.
def ref_max_pool3d(input, kernel_size, stride, padding, dilation, ceil_mode):
    # Here we use PyTorch's built-in MaxPool3d layer
    pool = torch.nn.MaxPool3d(kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, ceil_mode=ceil_mode)
    return pool(input)

# Test 1: Non-Cubic (Heterogeneous) Pooling Parameters
# Many general situations require per-dimension parameters.
# Our CUDA kernel always uses a single int for kernel_size (and related pooling offsets)
# so if a user wished to use non-cubic pooling (e.g., different kernel dimensions for D, H, W)
# the kernel would be wrong.
def test_cubic_requirement(cuda_kernel):
    # Even though PyTorch's MaxPool3d supports tuple parameters,
    # our extension only supports ints. Simulate what a user might try to do.
    # Here we assume a non-cubic case is desired: kernel_depth=3, kernel_height=2, kernel_width=2.
    # Instead, we pass kernel_size=3 (so the kernel will do 3x3x3) and a stride value that does not match.
    batch_size, channels, d, h, w = 1, 1, 8, 8, 8
    x = torch.randn(batch_size, channels, d, h, w, device="cuda", dtype=torch.float32)
    
    # Use non-cubic desired parameters for the reference and for the extension we pass the first value only.
    # With the built–in pooling, using tuple parameters this would yield a different result.
    ref_output = torch.nn.functional.max_pool3d(x, kernel_size=(3,2,2), stride=(2,1,1),
                                                 padding=1, dilation=1, ceil_mode=False)
    
    # Our kernel call (only supports int parameters)
    ext_output = cuda_kernel.forward(x, 3, 2, 1, 1, False, False)
    
    # Compare against the reference. They will differ because our kernel always does a cubic window.
    # We expect the maximum values to be different.
    torch.cuda.synchronize()
    max_diff = (ext_output - ref_output).abs().max().item()
    assert max_diff > 1e-3, f"Expected a mismatch because of cubic vs non-cubic pooling but got max_diff={max_diff}"

# Test 2: Concurrent Parameter Updates / Constant Memory Issue.
# Launch two separate calls sequentially with different parameters.
# If __constant__ memory is not handled correctly, the parameters from the first call may “leak” into the second.
def test_concurrent_parameters(cuda_kernel):
    batch_size, channels, d, h, w = 2, 2, 16, 16, 16
    x = torch.randn(batch_size, channels, d, h, w, device="cuda", dtype=torch.float32)
    
    # First pooling call with one set of parameters
    out1 = cuda_kernel.forward(x, 3, 2, 1, 1, False, False)
    ref1 = ref_max_pool3d(x, 3, 2, 1, 1, False)
    
    # Then immediately a second call with different parameters
    # Parameters are different; if __constant__ memory was not updated correctly, the wrong behavior may occur.
    out2 = cuda_kernel.forward(x, 2, 1, 0, 1, False, False)
    ref2 = ref_max_pool3d(x, 2, 1, 0, 1, False)
    
    torch.cuda.synchronize()
    
    # We expect the outputs to match the reference.
    err1 = (out1 - ref1).abs().max().item()
    err2 = (out2 - ref2).abs().max().item()
    
    # In a correct implementation both errors would be very small.
    # In our faulty implementation, one or both will be unacceptably large.
    assert err1 < 1e-3 or err2 < 1e-3, "At least one of the concurrent kernel calls did not match the reference output. Constant memory might be contaminated."

# Test 3: Non-Contiguous Input Error
# The implementation of the kernel uses simple index arithmetic assuming a contiguous input.
# When a non-contiguous tensor is passed, the output produced by our extension will be wrong.
def test_non_contiguous_input(cuda_kernel):
    batch_size, channels, d, h, w = 2, 2, 16, 16, 16
    # Create a contiguous tensor and then transpose to make it non-contiguous.
    x = torch.randn(batch_size, channels, d, h, w, device="cuda", dtype=torch.float32)
    x = x.transpose(2, 4)  # Now the tensor is not contiguous.
    assert not x.is_contiguous(), "Input tensor is expected to be non-contiguous for this test."
    
    # Compute the reference using PyTorch's built-in pooling which supports non-contiguous inputs.
    ref_output = ref_max_pool3d(x, 3, 2, 1, 1, False)
    ext_output = cuda_kernel.forward(x, 3, 2, 1, 1, False, False)
    
    torch.cuda.synchronize()
    err = (ext_output - ref_output).abs().max().item()
    assert err > 1e-3, f"Kernel should fail on non-contiguous input. Observed error={err} (output looks too similar to reference)."
