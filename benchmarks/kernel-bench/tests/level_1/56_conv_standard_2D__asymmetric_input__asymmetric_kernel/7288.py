
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# Helper to compile the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="custom_conv2d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: The kernel only supports float32.
# Test: Try running the kernel with float64 (double) input tensors.
def test_non_float32_input():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    
    # build extension kernel
    cuda_mod = build_kernel()
    
    # Create input, weight, bias as double, even though the kernel uses float pointers.
    batch_size = 2
    in_channels = 3
    out_channels = 4
    H_in, W_in = 16, 16
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = (1, 1)
    dilation = (1, 1)
    groups = 1
    
    # Create double precision tensors
    input_tensor = torch.randn(batch_size, in_channels, H_in, W_in, dtype=torch.float64, device='cuda')
    # The weight should have shape (out_channels, in_channels // groups, kH, kW).
    weight_tensor = torch.randn(out_channels, in_channels // groups, *kernel_size, dtype=torch.float64, device='cuda')
    bias_tensor = torch.randn(out_channels, dtype=torch.float64, device='cuda')
    
    # Run the custom kernel (which will interpret the data as float32)
    out_custom = cuda_mod.forward(input_tensor, weight_tensor, bias_tensor,
                                  list(stride), list(padding), list(dilation), groups)
    
    # Run the reference convolution using PyTorch (after converting to float32)
    input32 = input_tensor.float()
    weight32 = weight_tensor.float()
    bias32   = bias_tensor.float()
    out_ref = F.conv2d(input32, weight32, bias32, stride, padding, dilation, groups)

    # The custom kernel output will be computed incorrectly from the misinterpreted data.
    # We check that the results are not close.
    assert not torch.allclose(out_custom, out_ref, atol=1e-5), \
           "Test failed: kernel output for double input appears to match reference, " \
           "but the kernel should only support float32."

# Issue 2: The kernel assumes that channels are evenly divisible by groups.
# Test: Use a convolution configuration where the number of input or output channels
# is not exactly divisible by groups to trigger an out‐of‐range index / incorrect computation.
def test_groups_not_divisible():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    
    # build extension kernel
    cuda_mod = build_kernel()
    
    # Choose a configuration where the number of channels is not evenly divisible by groups.
    batch_size = 1
    in_channels = 3   # not evenly divisible by groups=2
    out_channels = 4  # assume weight shape is (4, in_channels//groups, kH, kW)
    H_in, W_in = 20, 20
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = (1, 1)
    dilation = (1, 1)
    groups = 2  # 3 // 2 == 1 (integer division), but then later group index computed on output channels becomes erroneous.
    
    # Create tensors of proper type (float32)
    input_tensor = torch.randn(batch_size, in_channels, H_in, W_in, dtype=torch.float32, device='cuda')
    # Since in_channels is 3 and groups=2, PyTorch’s Conv2d would normally reject this.
    # However, the custom kernel will use int division: in_channels//groups == 1, so weight shape is (4, 1, 3, 3)
    weight_tensor = torch.randn(out_channels, in_channels // groups, *kernel_size, dtype=torch.float32, device='cuda')
    bias_tensor = torch.randn(out_channels, dtype=torch.float32, device='cuda')
    
    # Running the custom conv kernel with this configuration is expected to yield an incorrect result or possibly
    # trigger a CUDA error (e.g., out-of-bound access). We catch the RuntimeError if a CUDA error is raised.
    with pytest.raises(RuntimeError):
        out_custom = cuda_mod.forward(input_tensor, weight_tensor, bias_tensor,
                                      list(stride), list(padding), list(dilation), groups)
        # If no exception is raised, we further check that the result deviates from a manually computed alternative.
        # (Note: torch.nn.functional.conv2d would itself raise an error for mismatched groups. Therefore,
        #  the successful return of an output from the custom kernel is an indication of a problem.)
    
# To allow running the tests directly
if __name__ == "__main__":
    pytest.main([__file__])
