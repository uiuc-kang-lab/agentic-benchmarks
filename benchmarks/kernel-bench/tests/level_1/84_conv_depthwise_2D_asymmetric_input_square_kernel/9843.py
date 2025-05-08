
import pytest
import torch
from torch.utils.cpp_extension import load

# Utility function to compile and load the CUDA kernel module.
def build_kernel():
    return load(
        name="custom_depthwise_conv",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

# Issue 1: The kernel is hard‐coded to support only float32.
def test_dtype_support():
    cuda_module = build_kernel()
    # Create inputs as double tensors.
    batch_size = 1
    in_channels = 1
    kernel_size = 3
    height = 16
    width = 16

    # Create a weight tensor shaped for depthwise convolution: [in_channels, channels_per_group, k, k]
    # For depthwise conv in our PyTorch example, channels_per_group is normally 1.
    input_tensor = torch.randn(batch_size, in_channels, height, width, dtype=torch.float64, device='cuda')
    weight_tensor = torch.randn(in_channels, 1, kernel_size, kernel_size, dtype=torch.float64, device='cuda')
    bias_tensor = torch.randn(in_channels, dtype=torch.float64, device='cuda')
    
    # Expect a failure because the kernel only handles float*.
    with pytest.raises(RuntimeError):
        cuda_module.forward(input_tensor, weight_tensor, bias_tensor, 1, 0)

# Issue 2: The kernel maps batches and output channels to gridDim.z, which can exceed device limits.
def test_exceed_grid_dim_z():
    cuda_module = build_kernel()
    # Many output channels: set groups=1 so that out_channels = in_channels, then artificially increase in_channels.
    batch_size = 1
    # Choose a very high number of in_channels such that gridDim.z (batch_size * out_channels) exceeds typical limits.
    # Note: Many CUDA devices have gridDim.z limit around 65535.
    in_channels = 70000  
    kernel_size = 3
    height = 32
    width = 32
    
    input_tensor = torch.randn(batch_size, in_channels, height, width, dtype=torch.float32, device='cuda')
    # For depthwise conv: weight shape: [in_channels, 1, kernel_size, kernel_size]
    weight_tensor = torch.randn(in_channels, 1, kernel_size, kernel_size, dtype=torch.float32, device='cuda')
    bias_tensor = torch.randn(in_channels, dtype=torch.float32, device='cuda')
    
    # Expect a CUDA launch error due to an excessive gridDim.z
    with pytest.raises(RuntimeError):
        cuda_module.forward(input_tensor, weight_tensor, bias_tensor, 1, 0)

# Issue 3: The kernel is hard-coded for kernel_size==3 in its unrolling branch.
def test_non_standard_kernel_size():
    cuda_module = build_kernel()
    batch_size = 1
    in_channels = 1
    # Use a kernel size other than 3 to force the non-unrolled branch.
    kernel_size = 5
    height = 20
    width = 20

    input_tensor = torch.randn(batch_size, in_channels, height, width, dtype=torch.float32, device='cuda')
    weight_tensor = torch.randn(in_channels, 1, kernel_size, kernel_size, dtype=torch.float32, device='cuda')
    bias_tensor = torch.randn(in_channels, dtype=torch.float32, device='cuda')
    
    # Even though this kernel branch is supported via pragma unroll,
    # if there is any issue with generality the output may be wrong.
    # Here we simply check that it runs without error.
    output = cuda_module.forward(input_tensor, weight_tensor, bias_tensor, 1, 0)
    assert output is not None

# Issue 4: The kernel assumes out_channels == in_channels * channels_per_group exactly.
def test_incorrect_channel_grouping():
    cuda_module = build_kernel()
    batch_size = 1
    in_channels = 4
    # Let channels_per_group be different from expected (simulate a group conv case)
    channels_per_group = 2
    # Therefore, expected out_channels would be 4*2 = 8. We purposely provide weight
    # with a mismatching shape.
    kernel_size = 3
    height = 16
    width = 16
    
    input_tensor = torch.randn(batch_size, in_channels, height, width, dtype=torch.float32, device='cuda')
    # Incorrect weight shape: using channels_per_group=1 for each channel.
    weight_tensor = torch.randn(in_channels, 1, kernel_size, kernel_size, dtype=torch.float32, device='cuda')
    bias_tensor = torch.randn(in_channels, dtype=torch.float32, device='cuda')
    
    # The forward() in our kernel expects weight.size(1)==channels_per_group.
    # This should trigger an index error or a TORCH_CHECK failure.
    with pytest.raises(RuntimeError):
        cuda_module.forward(input_tensor, weight_tensor, bias_tensor, 1, 0)

# Issue 5: The forward function requires contiguous tensors.
def test_non_contiguous_input():
    cuda_module = build_kernel()
    batch_size = 1
    in_channels = 1
    kernel_size = 3
    height = 16
    width = 16
    
    # Create a contiguous tensor and then create a non-contiguous view.
    input_tensor = torch.randn(batch_size, in_channels, height, width, dtype=torch.float32, device='cuda')
    input_tensor = input_tensor.transpose(2, 3)  # now non-contiguous
    weight_tensor = torch.randn(in_channels, 1, kernel_size, kernel_size, dtype=torch.float32, device='cuda')
    bias_tensor = torch.randn(in_channels, dtype=torch.float32, device='cuda')
    
    # Expect a failure because the forward() function requires contiguous tensors.
    with pytest.raises(RuntimeError):
        cuda_module.forward(input_tensor, weight_tensor, bias_tensor, 1, 0)
