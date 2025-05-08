
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    module = load(
        name="conv3d_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

# Issue 1: Non-uniform parameters (tuple stride, padding, dilation)
def test_non_uniform_parameters():
    # Build the kernel module
    conv3d_kernel = build_kernel()
    
    # Create a sample input that mimics a 5D tensor (N, C, D, H, W)
    batch_size = 2
    in_channels = 3
    depth = 10
    height = 10
    width = 10
    input_tensor = torch.randn(batch_size, in_channels, depth, height, width, device='cuda', dtype=torch.float32)
    
    # Create a weight tensor with shape (out_channels, in_channels/groups, kernel_d, kernel_h, kernel_w)
    out_channels = 4
    kernel_size = (3, 3, 3)
    weight = torch.randn(out_channels, in_channels, *kernel_size, device='cuda', dtype=torch.float32)
    
    # We use bias = None to simplify the test.
    # Try to pass non-scalar parameters. The binding expects int for stride, padding, and dilation.
    with pytest.raises(TypeError):
        _ = conv3d_kernel.forward(
            input_tensor,
            weight,
            None,
            stride=(1, 1, 1),    # should be an int not a tuple
            padding=(0, 0, 0),   # should be an int not a tuple
            dilation=(1, 1, 1),  # should be an int not a tuple
            groups=1
        )

# Issue 2: Data type limitation (only float32 is supported)
def test_incorrect_dtype():
    conv3d_kernel = build_kernel()
    
    # Create double precision (float64) input tensor and weight tensor
    batch_size = 2
    in_channels = 3
    depth = 8
    height = 8
    width = 8
    input_tensor = torch.randn(batch_size, in_channels, depth, height, width, device='cuda', dtype=torch.float64)
    
    out_channels = 4
    kernel_size = (3, 3, 3)
    weight = torch.randn(out_channels, in_channels, *kernel_size, device='cuda', dtype=torch.float64)
    
    # Using float64 tensors will cause the kernel to reinterpret memory (data_ptr<float>() is used),
    # leading to incorrect results. As a simple check, we compare the kernel output against PyTorch's conv3d.
    # Note: the kernel doesn't raise an error, but the output will be incorrect.
    output_kernel = conv3d_kernel.forward(
        input_tensor,
        weight,
        None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1
    )
    
    # Use native PyTorch conv3d which supports float64
    conv3d = torch.nn.Conv3d(in_channels, out_channels, kernel_size, bias=False).cuda().double()
    # Overwrite conv3d weights for comparison
    conv3d.weight.data.copy_(weight)
    output_ref = conv3d(input_tensor)
    
    # The outputs should drastically differ due to data type misinterpretation
    assert not torch.allclose(output_kernel, output_ref, atol=1e-5), \
        "Kernel incorrectly processed float64 input as if it were float32."

# Issue 3: Unnecessary inclusion of cuDNN headers is a code maintenance issue.
def test_unused_cudnn_includes():
    conv3d_kernel = build_kernel()
    
    # This test does not trigger a runtime error, but we simulate a check by verifying that the kernel works as expected.
    # If the cuDNN headers caused issues in the build, the module would fail to load.
    batch_size = 1
    in_channels = 1
    depth = 5
    height = 5
    width = 5
    input_tensor = torch.randn(batch_size, in_channels, depth, height, width, device='cuda', dtype=torch.float32)
    weight = torch.randn(1, in_channels, 3, 3, 3, device='cuda', dtype=torch.float32)
    
    output = conv3d_kernel.forward(
        input_tensor,
        weight,
        None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1
    )
    
    assert output is not None, "Output should not be None even if cuDNN is unused."

# Issue 4: Lack of CUDA kernel error checking
def test_kernel_launch_error_checking():
    conv3d_kernel = build_kernel()
    
    # Create an input tensor that is intentionally mismatched in dimensions to force an error.
    batch_size = 1
    in_channels = 3
    depth = 4
    height = 4
    width = 4
    input_tensor = torch.randn(batch_size, in_channels, depth, height, width, device='cuda', dtype=torch.float32)
    
    # Weight with incorrect kernel depth, height, or width causing out-of-bound accesses.
    # For example, if kernel size is larger than the input spatial dims.
    out_channels = 2
    kernel_size = (5, 5, 5)  # intentionally too large
    weight = torch.randn(out_channels, in_channels, *kernel_size, device='cuda', dtype=torch.float32)
    
    # The kernel loop will simply skip computations out-of-bound,
    # so instead we trigger an error by providing groups that do not divide channels correctly.
    with pytest.raises(RuntimeError):
        # This should cause unexpected behavior (group calculation error) that we expect to be caught.
        _ = conv3d_kernel.forward(
            input_tensor,
            weight,
            None,
            stride=1,
            padding=0,
            dilation=1,
            groups=2  # in_channels=3 is not divisible by groups=2
        )
