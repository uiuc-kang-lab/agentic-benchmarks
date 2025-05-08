
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Helper function to compile and load the CUDA extension.
def build_kernel():
    # Ensure that the source file "kernel.cu" exists in the current directory.
    assert os.path.exists("kernel.cu"), "kernel.cu not found in the current directory."
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1 & 2: Kernel width is forced to be 1.
def test_asymmetric_kernel_width_issue():
    # Create a dummy input
    batch_size = 2
    channels = 3
    in_h, in_w = 16, 16
    x = torch.randn(batch_size, channels, in_h, in_w, device="cuda", dtype=torch.float32)

    # Create a weight with kernel width > 1 (e.g. (channels, 1, kernel_h, kernel_w) with kernel_w != 1)
    # The kernel expects a weight shape of (channels, 1, kernel_h) flattened to constant memory.
    # Here we deliberately create a weight with an extra horizontal dimension.
    kernel_h, kernel_w = 3, 3  # asymmetric kernel that is not 1-wide.
    weight = torch.randn(channels, 1, kernel_h, kernel_w, device="cuda", dtype=torch.float32)
    
    bias = None
    stride = 1
    padding = 0
    dilation = 1
    groups = channels

    mod = build_kernel()

    # We expect the kernel to compute output using only the first column of the weight.
    # Hence, comparing its output against a conventional depthwise conv2d (which uses the full kernel)
    # should reveal a mismatch.
    output_custom = mod.forward(x, weight.squeeze(1)[:, :kernel_h], bias, stride, padding, dilation, groups)
    
    # Build a reference convolution using PyTorch's conv2d.
    conv_ref = torch.nn.Conv2d(
        channels, channels, kernel_size=(kernel_h, kernel_w), stride=stride,
        padding=padding, dilation=dilation, groups=channels, bias=False
    ).to("cuda")
    # Manually set the conv_ref weights to use the same values as in our "weight"
    # For a fair comparison, we zero out all columns except the first column.
    with torch.no_grad():
        ref_weight = weight.clone()
        ref_weight[..., 1:] = 0  # force horizontal kernel values except first column to be zero.
        conv_ref.weight.copy_(ref_weight)
    
    output_ref = conv_ref(x)
    # Expect the outputs to differ if the user had intended to use the full kernel.
    assert not torch.allclose(output_custom, output_ref, atol=1e-5), \
        "Test failed: Custom kernel output matches reference even though kernel width issue exists."

# Issue 3: Unrolling on a runtime parameter.
def test_dynamic_kernel_unroll():
    # We can trigger potential unrolling issues (e.g. excessive register usage)
    # by using a larger kernel height approaching the constant memory limit.
    batch_size = 1
    channels = 2  # keep channels low so that we don't trigger constant memory limit here.
    in_h, in_w = 64, 64
    kernel_h = 11  # maximum allowed, dynamic unrolling might behave poorly.
    x = torch.randn(batch_size, channels, in_h, in_w, device="cuda", dtype=torch.float32)
    # Create a weight with kernel width forced to be 1, as expected, but with a large kernel_h.
    weight = torch.randn(channels, 1, kernel_h, device="cuda", dtype=torch.float32)
    bias = None
    stride = 1
    padding = 0
    dilation = 1
    groups = channels

    mod = build_kernel()
    # We do not have a reference to compare to.
    # Instead, we run the kernel and check for CUDA errors after execution.
    output = mod.forward(x, weight.squeeze(1), bias, stride, padding, dilation, groups)
    torch.cuda.synchronize()
    # If excessive unrolling were to cause issues (e.g., crashes or mis-computation),
    # the above line would trigger a CUDA error.
    assert output is not None, "Kernel execution failed with dynamic unroll parameters."

# Issue 4: Kernel only supports float32.
def test_input_dtype_issue():
    batch_size = 2
    channels = 3
    in_h, in_w = 32, 32
    # Use float64 input to trigger dtype issue.
    x = torch.randn(batch_size, channels, in_h, in_w, device="cuda", dtype=torch.float64)
    kernel_h = 3
    weight = torch.randn(channels, 1, kernel_h, device="cuda", dtype=torch.float64)
    bias = None
    stride = 1
    padding = 0
    dilation = 1
    groups = channels

    mod = build_kernel()
    with pytest.raises(RuntimeError):
        # Expecting a runtime error because the kernel and constant memory copies expect float32.
        mod.forward(x, weight.squeeze(1), bias, stride, padding, dilation, groups)
    torch.cuda.synchronize()

# Issue 5: No error check on cudaMemcpyToSymbol.
def test_constant_memory_copy_error():
    # This test attempts to trigger an error in cudaMemcpyToSymbol by exceeding constant memory limits.
    batch_size = 1
    # Set channels to exceed MAX_CHANNELS defined in the CUDA code (which is 256)
    channels = 300  
    in_h, in_w = 32, 32
    kernel_h = 3
    x = torch.randn(batch_size, channels, in_h, in_w, device="cuda", dtype=torch.float32)
    weight = torch.randn(channels, 1, kernel_h, device="cuda", dtype=torch.float32)
    bias = None
    stride = 1
    padding = 0
    dilation = 1
    groups = channels

    mod = build_kernel()
    with pytest.raises(RuntimeError):
        # The forward function should throw a runtime error because channels exceed MAX_CHANNELS.
        mod.forward(x, weight.squeeze(1), bias, stride, padding, dilation, groups)
    torch.cuda.synchronize()

# Issue 6: Horizontal dilation is not supported.
def test_horizontal_dilation_issue():
    # Here we simulate a scenario where dilation is not 1.
    # Since the kernel only adjusts vertical indexing based on dilation, wrong output is expected
    # when a dilation value is applied.
    batch_size = 1
    channels = 2
    in_h, in_w = 20, 20
    kernel_h = 3
    x = torch.randn(batch_size, channels, in_h, in_w, device="cuda", dtype=torch.float32)
    weight = torch.randn(channels, 1, kernel_h, device="cuda", dtype=torch.float32)
    bias = None
    stride = 1
    padding = 1
    dilation = 2  # horizontal dilation is set, but kernel ignores horizontal dilation.
    groups = channels

    mod = build_kernel()
    output_custom = mod.forward(x, weight.squeeze(1), bias, stride, padding, dilation, groups)
    
    # Build a reference convolution using PyTorch that correctly considers dilation in both dimensions.
    conv_ref = torch.nn.Conv2d(
        channels, channels, kernel_size=(kernel_h, 1), stride=stride,
        padding=padding, dilation=dilation, groups=channels, bias=False
    ).to("cuda")
    with torch.no_grad():
        conv_ref.weight.copy_(weight)
        
    output_ref = conv_ref(x)
    # Because the custom kernel does not apply horizontal dilation,
    # the output should differ from the PyTorch reference output.
    assert not torch.allclose(output_custom, output_ref, atol=1e-5), \
        "Test failed: Custom kernel output unintentionally matched reference output despite horizontal dilation issue."
