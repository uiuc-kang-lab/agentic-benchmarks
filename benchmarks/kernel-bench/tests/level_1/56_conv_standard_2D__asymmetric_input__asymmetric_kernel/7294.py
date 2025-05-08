
import pytest
import torch
from torch.utils.cpp_extension import load

# Utility to (re)build the CUDA extension from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="custom_conv2d",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# A helper function to perform a reference convolution using PyTorch's nn.functional.conv2d.
def reference_conv2d(input, weight, bias, stride, padding, dilation, groups):
    return torch.nn.functional.conv2d(input, weight, bias=bias, stride=stride,
                                       padding=padding, dilation=dilation, groups=groups)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_non_float32_dtype():
    """
    Issue 1: Pass double precision inputs.
    The kernel is compiled for float32 and uses data_ptr<float>().
    When double inputs are passed, the kernel will produce incorrect output.
    """
    cuda_mod = build_kernel()
    # Create double precision inputs.
    N, C_in, H, W = 2, 3, 16, 16
    C_out, K_h, K_w = 4, 3, 3
    input = torch.randn(N, C_in, H, W, device="cuda", dtype=torch.double)
    weight = torch.randn(C_out, C_in, K_h, K_w, device="cuda", dtype=torch.double)
    bias = torch.randn(C_out, device="cuda", dtype=torch.double)
    stride = [1, 1]
    padding = [1, 1]
    dilation = [1, 1]
    groups = 1

    # Cast expected reference result using conv2d (which supports double)
    ref = torch.nn.functional.conv2d(input, weight, bias, stride, padding, dilation, groups)
    
    # Call kernel; since the kernel reads floats, we expect a mismatch
    out = cuda_mod.forward(input, weight, bias, stride, padding, dilation, groups)
    torch.cuda.synchronize()

    # The mismatch is expected when using double input.
    with pytest.raises(AssertionError):
        assert torch.allclose(out, ref, atol=1e-4), "Kernel produced correct output on double dtype!"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_invalid_group_division():
    """
    Issue 2: Use groups that do not evenly divide the number of output channels.
    The kernel’s group computation leads to incorrect indexing.
    """
    cuda_mod = build_kernel()
    # Setup parameters where C_out is not divisible by groups.
    # Standard PyTorch conv2d requires that C_in and C_out are divisible by groups,
    # but our custom kernel does not validate it.
    N, C_in, H, W = 2, 4, 16, 16
    C_out, K_h, K_w = 5, 3, 3  # 5 is not divisible by groups=2
    stride = [1, 1]
    padding = [1, 1]
    dilation = [1, 1]
    groups = 2

    input = torch.randn(N, C_in, H, W, device="cuda", dtype=torch.float32)
    weight = torch.randn(C_out, C_in // groups, K_h, K_w, device="cuda", dtype=torch.float32)
    bias = torch.randn(C_out, device="cuda", dtype=torch.float32)

    ref = torch.nn.functional.conv2d(input, weight, bias, stride, padding, dilation, groups)
    out = cuda_mod.forward(input, weight, bias, stride, padding, dilation, groups)
    torch.cuda.synchronize()

    with pytest.raises(AssertionError):
        # Expecting a mismatch because the kernel's group logic is wrong in this case.
        assert torch.allclose(out, ref, atol=1e-4), "Kernel produced correct output with invalid groups division!"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_warp_size_assumption():
    """
    Issue 3: The kernel assumes a warp size of 32.
    If the number of output elements does not align to a multiple of 32 
    the indexing may be fragile on devices with a different warp size.
    We simulate this scenario with an output element count not divisible by 32.
    """
    cuda_mod = build_kernel()
    # Setup parameters such that total output elements (N * C_out * H_out * W_out) is not a multiple of 32.
    N, C_in, H, W = 1, 3, 17, 17  # arbitrary sizes to yield a total output count not divisible by 32.
    C_out, K_h, K_w = 7, 3, 3  # 7 output channels (with groups=1, valid, but arbitrary)
    stride = [1, 1]
    padding = [1, 1]
    dilation = [1, 1]
    groups = 1

    input = torch.randn(N, C_in, H, W, device="cuda", dtype=torch.float32)
    weight = torch.randn(C_out, C_in, K_h, K_w, device="cuda", dtype=torch.float32)
    bias = None

    ref = torch.nn.functional.conv2d(input, weight, bias, stride, padding, dilation, groups)
    out = cuda_mod.forward(input, weight, bias, stride, padding, dilation, groups)
    torch.cuda.synchronize()

    # Although the kernel does the boundary check, the hardcoded warp size assumption 
    # may lead to errors on devices with a different native warp size than 32.
    # We force a failure if the output does not approximately match the PyTorch reference.
    with pytest.raises(AssertionError):
        assert torch.allclose(out, ref, atol=1e-4), "Kernel output matches reference despite potential warp-size issues!"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_unused_helper_function():
    """
    Issue 4: The helper function is_valid_input_pos is unused.
    While this is not a runtime error, this test ensures that the code does not utilize it,
    which might be a sign of incomplete implementation for more general situations.
    This test dynamically inspects the kernel source to detect unused functions.
    """
    with open("kernel.cu", "r") as f:
        source = f.read()

    # Check that is_valid_input_pos is present but not called inside conv2d_cuda_kernel_warp_optimized.
    assert "is_valid_input_pos" in source, "Helper function not found in source!"
    # A crude check: the kernel function should not call is_valid_input_pos.
    assert "is_valid_input_pos(" not in source.split("conv2d_cuda_kernel_warp_optimized")[1], \
        "Kernel is using the helper function, which is unexpected!"

