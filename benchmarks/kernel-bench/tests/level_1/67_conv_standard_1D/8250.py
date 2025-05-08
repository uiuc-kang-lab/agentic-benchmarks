
import os
import tempfile
import torch
import pytest
from torch.utils.cpp_extension import load

# Helper function to build the CUDA extension from a given source file path.
def build_kernel(source_path, extra_cuda_cflags=None):
    module = load(
        name="test_conv1d_kernel",
        sources=[source_path],
        extra_cuda_cflags=(extra_cuda_cflags or []),
        verbose=True,
    )
    return module

# -----------------------------------------------------------------------------
# Test case for Issue 1:
#   Trigger the bug when the launch configuration does not use a thread count 
#   that is a multiple of 64. We simulate this by modifying the kernel source to 
#   use a block size different from 256 (e.g., 70 threads per block) when launching,
#   which breaks the assumption about "threads per output" division.
# -----------------------------------------------------------------------------
def test_incorrect_block_size(tmp_path):
    # Read the original kernel source
    kernel_file = os.path.join(os.path.dirname(__file__), "kernel.cu")
    with open(kernel_file, "r") as f:
        src = f.read()
        
    # Modify the launch configuration line inside conv1d_forward_impl:
    # Replace "int threads_per_block = 256;" with our modified block size.
    # (Note: in a real scenario, the block size should be configurable.
    #  For this test we simulate a wrong configuration.)
    bad_block_size = 70
    modified_src = src.replace("int threads_per_block = 256;", f"int threads_per_block = {bad_block_size};")
    
    # Write the modified source to a temporary file.
    temp_kernel = tmp_path / "kernel_bad_block.cu"
    temp_kernel.write_text(modified_src)
    
    # Build the kernel with the modified source.
    conv_module = build_kernel(str(temp_kernel))
    
    # Create a simple test input that would normally work.
    batch_size = 1
    in_channels = 4
    out_channels = 8
    kernel_size = 3
    length = 16
    stride = 1
    padding = 1
    dilation = 1
    groups = 1
    
    # Create input, weight and bias (bias is optional)
    x = torch.randn(batch_size, in_channels, length, device="cuda", dtype=torch.float32)
    # weight shape: [C_out, C_in/groups, kernel_size]
    weight = torch.randn(out_channels, in_channels // groups, kernel_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float32)
    
    # Run the custom CUDA kernel.
    y = conv_module.forward(x, weight, bias, stride, padding, dilation, groups)
    
    # For a correct implementation (with proper block size), we use PyTorch's functional conv1d.
    # Note: nn.functional.conv1d expects weight of shape [C_out, C_in, kernel_size] when groups==1.
    # Here we adapt our weight shape accordingly.
    weight_ref = weight
    y_ref = torch.nn.functional.conv1d(x, weight_ref, bias, stride, padding, dilation, groups)
    
    # The outputs should match if the launch configuration were correct.
    # With a bad block size the cooperative reduction is done wrongly.
    # We expect the maximum absolute error to be larger than an acceptable tolerance.
    max_error = (y - y_ref).abs().max().item()
    # Set a small tolerance under normal conditions.
    tol = 1e-4
    assert max_error > tol, (
        f"Test failed to trigger issue1: max_error ({max_error}) did not exceed tolerance ({tol}). "
        "Kernel assumed a correct launch config even though block size is not a multiple of 64."
    )

# -----------------------------------------------------------------------------
# Test case for Issue 2:
#   Use an unsupported data type (fp16) as input. The kernel is implemented solely
#   for float32 and should throw an error (via TORCH_CHECK) when non-float32 tensors
#   are provided.
# -----------------------------------------------------------------------------
def test_unsupported_dtype(tmp_path):
    # Build the kernel normally.
    kernel_file = os.path.join(os.path.dirname(__file__), "kernel.cu")
    conv_module = build_kernel(kernel_file)
    
    batch_size = 1
    in_channels = 4
    out_channels = 8
    kernel_size = 3
    length = 16
    stride = 1
    padding = 1
    dilation = 1
    groups = 1
    
    # Create fp16 input, weight and bias.
    x = torch.randn(batch_size, in_channels, length, device="cuda", dtype=torch.float16)
    weight = torch.randn(out_channels, in_channels // groups, kernel_size, device="cuda", dtype=torch.float16)
    bias = torch.randn(out_channels, device="cuda", dtype=torch.float16)
    
    # Expect an error because the kernel does not support f16;
    # The C++ code checks for at::kFloat (i.e. float32) and should trigger a TORCH_CHECK failure.
    with pytest.raises(RuntimeError) as excinfo:
        conv_module.forward(x, weight, bias, stride, padding, dilation, groups)
    assert "float32" in str(excinfo.value), (
        "Kernel did not report an error when using unsupported dtype (fp16)."
    )
    
if __name__ == "__main__":
    pytest.main([__file__])
