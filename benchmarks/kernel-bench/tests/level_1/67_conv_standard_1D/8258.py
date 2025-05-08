
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    # Load the CUDA extension from kernel.cu; assume it's in the current directory.
    cuda_module = load(
        name="conv1d_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
    return cuda_module

# Helper: build a mini function to call the kernel's forward.
def run_conv1d_forward(x, weight, bias, stride, padding, dilation, groups):
    module = build_kernel()
    # Convert bias to pybind11 None if not provided
    bias_obj = bias if bias is not None else None
    return module.forward(x, weight, bias_obj, stride, padding, dilation, groups)

# Issue 1: Non-contiguous tensors are not supported.
def test_non_contiguous_input():
    batch_size = 8
    in_channels = 4
    out_channels = 8
    kernel_size = 3
    length = 128
    stride = 1
    padding = 1
    dilation = 1
    groups = 1

    # Create contiguous input and weight tensors
    x = torch.randn(batch_size, in_channels, length, device="cuda", dtype=torch.float32)
    weight = torch.randn(out_channels, in_channels // groups, kernel_size, device="cuda", dtype=torch.float32)
    bias = None

    # Make a non-contiguous version by transposing the last two dimensions and transposing back.
    x_noncontig = x.transpose(1,2).transpose(1,2)
    # Assert that the underlying tensor is not contiguous
    assert not x_noncontig.is_contiguous(), "Test setup error: x_noncontig is contiguous."

    # Expect the kernel to fail or produce incorrect result.
    with pytest.raises(RuntimeError):
        # The kernel may either crash or raise an error because it expects contiguous memory.
        y = run_conv1d_forward(x_noncontig, weight, bias, stride, padding, dilation, groups)
        torch.cuda.synchronize()

# Issue 2: The kernel only supports float32. Passing a different dtype should trigger a TORCH_CHECK failure.
def test_dtype_mismatch():
    batch_size = 8
    in_channels = 4
    out_channels = 8
    kernel_size = 3
    length = 128
    stride = 1
    padding = 1
    dilation = 1
    groups = 1

    # Create double precision tensors.
    x = torch.randn(batch_size, in_channels, length, device="cuda", dtype=torch.float64)
    weight = torch.randn(out_channels, in_channels // groups, kernel_size, device="cuda", dtype=torch.float64)
    bias = None

    with pytest.raises(RuntimeError):
        # The extension uses TORCH_CHECK to enforce float32 so this call should fail.
        y = run_conv1d_forward(x, weight, bias, stride, padding, dilation, groups)
        torch.cuda.synchronize()

# Issue 3: Dimension mismatch with groups.
def test_incorrect_dimensions_for_groups():
    batch_size = 8
    in_channels = 5  # not divisible by groups (e.g., groups = 2)
    out_channels = 8
    kernel_size = 3
    length = 128
    stride = 1
    padding = 1
    dilation = 1
    groups = 2  # Since 5 is not divisible by 2, the kernel indexing into weight will be wrong.

    x = torch.randn(batch_size, in_channels, length, device="cuda", dtype=torch.float32)
    # Weight shape is expected to be [C_out, C_in/groups, kernel_size]; here C_in//groups would be 5//2 = 2 (truncated)
    # which does not match the intended channel division.
    weight = torch.randn(out_channels, in_channels // groups, kernel_size, device="cuda", dtype=torch.float32)
    bias = None

    with pytest.raises(RuntimeError):
        # Expect a runtime error due to inconsistency in parameters.
        y = run_conv1d_forward(x, weight, bias, stride, padding, dilation, groups)
        torch.cuda.synchronize()
        
if __name__ == "__main__":
    pytest.main([__file__])
