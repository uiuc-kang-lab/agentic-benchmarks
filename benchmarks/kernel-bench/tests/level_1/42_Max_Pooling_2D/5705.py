
import torch
import pytest
from torch.utils.cpp_extension import load
import torch.nn.functional as F

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test case for issue 1 and issue 3:
# When kernel_size is not 2 or 3 (e.g. 4), the kernel instantiation is incorrect.
def test_arbitrary_kernel_size():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    cuda_module = build_kernel()
    # Use a kernel_size that is not 2 or 3 to force the fallback branch.
    kernel_size = 4
    stride = 2
    padding = 1
    dilation = 1

    # Create a simple input tensor.
    batch_size = 2
    channels = 3
    height = 16
    width = 16
    input_tensor = torch.randn(batch_size, channels, height, width, device="cuda", dtype=torch.float32)

    # Run the custom CUDA kernel function.
    try:
        output_custom = cuda_module.forward(input_tensor, kernel_size, stride, padding, dilation)
    except Exception as e:
        pytest.fail(f"Kernel launch failed with kernel_size={kernel_size}: {e}")

    # Calculate reference output using PyTorch's built-in MaxPool2d.
    ref_layer = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    output_ref = ref_layer(input_tensor)

    # We expect a discrepancy because the kernel is miscompiled for arbitrary kernel_size.
    # Thus, the outputs should not match. If they match, we didn't trigger the issue.
    if torch.allclose(output_custom, output_ref, atol=1e-5):
        pytest.fail("Arbitrary kernel_size (not 2 or 3) did not trigger an error; expected wrong behavior due to template mis-instantiation.")

# Test case for issue 2:
# When input of type double is used, the use of fmaxf is invalid.
def test_double_precision():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    cuda_module = build_kernel()
    kernel_size = 2  # Use one of the supported sizes so that the code path is taken.
    stride = 2
    padding = 1
    dilation = 1

    batch_size = 2
    channels = 3
    height = 16
    width = 16
    # Create a double precision input tensor.
    input_tensor = torch.randn(batch_size, channels, height, width, device="cuda", dtype=torch.float64)

    # Run the custom CUDA kernel function.
    try:
        output_custom = cuda_module.forward(input_tensor, kernel_size, stride, padding, dilation)
    except Exception as e:
        pytest.fail(f"Kernel launch failed with double precision input: {e}")

    # Calculate reference output using PyTorch's built-in MaxPool2d.
    ref_layer = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    output_ref = ref_layer(input_tensor)

    # The results should be very different if the wrong function (fmaxf) is used;
    # if they are close, then issue 2 is not triggered.
    if torch.allclose(output_custom, output_ref, atol=1e-5):
        pytest.fail("Double precision input did not trigger an error; fmaxf seems to work unexpectedly for double.")
