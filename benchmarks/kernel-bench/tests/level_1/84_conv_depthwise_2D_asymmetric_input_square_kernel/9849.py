
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper function to build the module from kernel.cu
def build_kernel():
    cuda_module = load(
        name="depthwise_conv_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1: Kernel assumes float32
def test_dtype_mismatch():
    # Build the module
    mod = build_kernel()
    
    # Create inputs in double precision (float64)
    batch_size = 4
    in_channels = 3
    kernel_size = 3
    height_in, width_in = 16, 16
    stride = 1
    padding = 0

    # Weight layout for depthwise convolution: [in_channels, channels_per_group, k, k].
    # For depthwise conv groups == in_channels so channels_per_group is usually 1.
    weight = torch.randn(in_channels, 1, kernel_size, kernel_size, dtype=torch.float64, device='cuda')
    # Bias if needed
    bias = torch.randn(in_channels, dtype=torch.float64, device='cuda')
    input_tensor = torch.randn(batch_size, in_channels, height_in, width_in, dtype=torch.float64, device='cuda')
    
    # Expect the kernel checks (or later memory errors) because the code assumes float32.
    with pytest.raises(RuntimeError):
        # This should trigger a failure (or an assertion inside the CUDA kernel)
        mod.forward(input_tensor, weight, bias, stride, padding)

# Issue 2: Kernel is tailored for depthwise convolution only.
def test_incorrect_weight_shape():
    # Build the module
    mod = build_kernel()
    
    # Create inputs with float32 but with a wrong weight shape. For a depthwise conv,
    # the weight shape is expected to be (in_channels, 1, kernel_size, kernel_size).
    # Here we use a weight with a second dimension not equal to 1 (e.g. a standard conv weight).
    batch_size = 2
    in_channels = 3
    # For standard convolution, one might have out_channels != in_channels and weight shape:
    # (out_channels, in_channels, kernel_size, kernel_size). We deliberately give a wrong shape.
    out_channels = 3  # not depthwise since groups will be taken as in_channels (3)
    kernel_size = 3
    height_in, width_in = 20, 20
    stride = 1
    padding = 0

    # Create an input tensor
    input_tensor = torch.randn(batch_size, in_channels, height_in, width_in, device='cuda', dtype=torch.float32)
    
    # Create a weight tensor that is not in the expected layout for a depthwise conv.
    # Expected layout for depthwise would be (in_channels, 1, kernel_size, kernel_size).
    # Instead, we pass a standard conv weight shape (out_channels, in_channels, kernel_size, kernel_size).
    weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device='cuda', dtype=torch.float32)
    
    # Create a bias tensor that matches out_channels (this is what the PyTorch model expects in standard conv)
    bias = torch.randn(out_channels, device='cuda', dtype=torch.float32)

    # The forward function in the extension will compute channels_per_group = weight.size(1)
    # and then out_channels = in_channels * channels_per_group.
    # In this test, channels_per_group = in_channels and so computed out_channels = in_channels * in_channels,
    # which is not equal to our provided out_channels. We then expect a size mismatch check to trigger.
    with pytest.raises(RuntimeError):
        mod.forward(input_tensor, weight, bias, stride, padding)
