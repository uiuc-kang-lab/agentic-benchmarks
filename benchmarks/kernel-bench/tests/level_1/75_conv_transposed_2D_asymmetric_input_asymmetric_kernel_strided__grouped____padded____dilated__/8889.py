
import pytest
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

# Utility function to build the extension module from kernel.cu
def build_kernel():
    cuda_module = load(
        name="conv_transpose2d_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Test case 1: Triggering the offset computation issue.
# We use a configuration in which the height offset search fails.
# For example, when stride_h == 2 and dilation_h == 2, consider:
#   oh=0, pad_h=1  => candidate_h = 1, mod_h = 1.
#   Loop i=0 gives (0*2)%2 = 0 and i=1 gives (1*2)%2 = 0.
# So no i satisfies the condition and offset_kh remains -1.
#
# With these parameters the output of our custom kernel will differ 
# from the output of nn.ConvTranspose2d.
def test_offset_issue():
    torch.manual_seed(42)
    batch = 2
    in_channels = 4
    out_channels = 4
    kernel_size = (3, 3)
    # Use a configuration to trigger offset_kh failure:
    stride = (2, 1)         # Here, stride_h is 2.
    padding = (1, 1)
    dilation = (2, 1)       # Here, dilation_h is 2.
    groups = 1
    bias_flag = False

    # Create input
    in_h, in_w = 8, 8
    x = torch.randn(batch, in_channels, in_h, in_w, device="cuda", dtype=torch.float32)
    
    # Build a ConvTranspose2d layer from PyTorch as reference.
    conv_ref = nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding, dilation=dilation,
        groups=groups, bias=bias_flag
    ).to("cuda")
    
    # Get reference output
    out_ref = conv_ref(x)
    
    # Prepare weight and bias for the custom CUDA kernel.
    # The weight in PyTorch for ConvTranspose2d has the shape:
    # [in_channels, out_channels/groups, kernel_h, kernel_w]
    weight = conv_ref.weight.detach().clone()
    bias_tensor = conv_ref.bias.detach().clone() if conv_ref.bias is not None else None

    # Build our custom kernel module.
    cuda_module = build_kernel()
    
    # Call the custom kernel forward.
    out_cuda = cuda_module.forward(
        x, weight, bias_tensor,
        list(stride), list(padding), list(dilation), groups
    )
    torch.cuda.synchronize()
    
    # Due to the offset issue, the outputs will differ. 
    # We check and assert that the maximum absolute difference is above a small tolerance.
    diff = (out_cuda - out_ref).abs().max().item()
    # The test is designed to trigger the issue so it should NOT be close.
    assert diff > 1e-3, f"Test did not trigger the offset issue (max diff {diff} too low)."

# Test case 2: Triggering the type assumption issue.
# The custom kernel always uses float pointers. Passing non-float32 tensors (e.g. float64)
# should lead to undefined behavior (and likely an error) in the kernel.
def test_dtype_issue():
    torch.manual_seed(123)
    batch = 1
    in_channels = 2
    out_channels = 2
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = (0, 0)
    dilation = (1, 1)
    groups = 1
    bias_flag = False

    in_h, in_w = 8, 8
    # Create double precision input to trigger the type issue.
    x = torch.randn(batch, in_channels, in_h, in_w, device="cuda", dtype=torch.float64)
    
    conv_ref = nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding, dilation=dilation,
        groups=groups, bias=bias_flag
    ).to("cuda").double()
    # Prepare reference weight and bias in double precision.
    weight = conv_ref.weight.detach().clone()
    bias_tensor = conv_ref.bias.detach().clone() if conv_ref.bias is not None else None

    cuda_module = build_kernel()
    
    # Expect the kernel to misbehave (likely crash or produce wrong results)
    with pytest.raises(Exception):
        # As our custom kernel always assumes float32, this should raise an error.
        out_cuda = cuda_module.forward(
            x, weight, bias_tensor,
            list(stride), list(padding), list(dilation), groups
        )
        torch.cuda.synchronize()

if __name__ == "__main__":
    pytest.main([__file__])
