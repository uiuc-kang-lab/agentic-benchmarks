
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper function to (re)build the module from kernel.cu.
def build_kernel():
    return load(
        name="test_conv1d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

# 1. Test to trigger the data type issue.
def test_dtype_issue():
    # Create input tensors of type double (float64) instead of float32.
    B = 2
    in_channels = 3
    in_size = 32
    out_channels = 4
    kernel_size = 3
    stride = 1
    dilation = 1

    # Tensors in double precision.
    x = torch.randn(B, in_channels, in_size, dtype=torch.double, device="cuda")
    weight = torch.randn(out_channels, in_channels, kernel_size, dtype=torch.double, device="cuda")
    bias = torch.randn(out_channels, dtype=torch.double, device="cuda")

    module = build_kernel()
    with pytest.raises((RuntimeError, AssertionError)):
        # The kernel expects float32 so the behavior is undefined.
        # Our wrapper in C++ does not check the dtype so index arithmetic will be wrong.
        # We check that the wrong dtype triggers a failure.
        module.forward(x, weight, bias, stride, dilation)

# 2. Test to trigger the fixed block size issue.
def test_fixed_block_size_issue():
    # Create a workload with a number of output elements that is not a multiple of 64.
    # This test verifies that the kernel launch works for odd sizes, but the fixed block size
    # may be suboptimal under such circumstances.
    B = 1
    in_channels = 2
    in_size = 17  # prime number size to force odd behavior in output size
    out_channels = 2
    kernel_size = 3
    stride = 2
    dilation = 1

    # Expected out_size: (17 - (1*(3-1)) - 1) // 2 + 1 = (17 - 2 - 1)//2 +1 = (14)//2 + 1 = 7 + 1 = 8 ? 
    # Actually: (17 - 2 - 1)//2 +1 = 14//2+1 = 7+1 = 8.
    x = torch.randn(B, in_channels, in_size, dtype=torch.float32, device="cuda")
    weight = torch.randn(out_channels, in_channels, kernel_size, dtype=torch.float32, device="cuda")
    bias = torch.randn(out_channels, dtype=torch.float32, device="cuda")

    module = build_kernel()
    output = module.forward(x, weight, bias, stride, dilation)
    # There is no exception thrown but we print the shape and do trivial check.
    expected_out_size = (in_size - dilation * (kernel_size - 1) - 1) // stride + 1
    assert output.shape == (B, out_channels, expected_out_size), \
        f"Output shape mismatch: got {output.shape} expected {(B, out_channels, expected_out_size)}"
    # Note: This test highlights that the fixed block size is used even when the workload does not align with it.
    
# 3. Test to trigger the weight layout issue.
def test_weight_layout_issue():
    # Create a weight tensor with an incorrect layout.
    B = 2
    in_channels = 3
    in_size = 32
    out_channels = 4
    kernel_size = 3
    stride = 1
    dilation = 1

    x = torch.randn(B, in_channels, in_size, dtype=torch.float32, device="cuda")
    # Instead of shape (out_channels, in_channels, kernel_size), we create a transposed weight:
    wrong_weight = torch.randn(in_channels, out_channels, kernel_size, dtype=torch.float32, device="cuda")
    # To pass the initial checks in the wrapper, we force wrong weight to be contiguous and 3D.
    wrong_weight = wrong_weight.contiguous()
    # Also supply a bias of correct size (out_channels).
    bias = torch.randn(out_channels, dtype=torch.float32, device="cuda")
    
    module = build_kernel()
    with pytest.raises(AssertionError):
        # The wrapper checks weight.size(1)==x.size(1). Here, wrong_weight.size(1)==out_channels and x.size(1)==in_channels.
        # This should trigger the "Input channels mismatch" check.
        module.forward(x, wrong_weight, bias, stride, dilation)

# 4. Test to check for lack of synchronization error detection.
def test_no_synchronization_issue():
    # Create a scenario where the result would be invalid if asynchronous errors are not caught.
    # We deliberately create an output tensor that when launched, if errors occur asynchronously,
    # they might not be reported immediately.
    # Here, we use dimensions that result in a very large number of threads.
    B = 32
    in_channels = 8
    in_size = 512
    out_channels = 16
    kernel_size = 5
    stride = 1
    dilation = 1

    x = torch.randn(B, in_channels, in_size, dtype=torch.float32, device="cuda")
    weight = torch.randn(out_channels, in_channels, kernel_size, dtype=torch.float32, device="cuda")
    bias = torch.randn(out_channels, dtype=torch.float32, device="cuda")
    
    module = build_kernel()
    # Launch the kernel.
    output = module.forward(x, weight, bias, stride, dilation)
    # Without calling torch.cuda.synchronize(), asynchronous errors might be hidden.
    # We force a synchronization here and assert the output has the right shape.
    torch.cuda.synchronize()
    expected_out_size = (in_size - dilation * (kernel_size - 1) - 1) // stride + 1
    assert output.shape == (B, out_channels, expected_out_size), \
        f"Output shape mismatch: got {output.shape}, expected {(B, out_channels, expected_out_size)}"
