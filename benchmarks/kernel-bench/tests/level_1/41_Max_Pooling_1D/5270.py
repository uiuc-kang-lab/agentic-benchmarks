
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the CUDA extension module from the kernel source file.
    return load(
        name="maxpool1d_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )

# Issue 1: The kernel only accepts float32 inputs.
def test_input_type_enforcement():
    # Create a double tensor (not float32).
    x = torch.randn(2, 3, 16, device="cuda", dtype=torch.double)
    module = build_kernel()
    # Expect the forward check to throw an error because the kernel expects float.
    with pytest.raises(RuntimeError):
        # Call the forward with double tensor. The kernel is hard-coded for float.
        module.forward(x, 3, 1, 0, 1, False)

# Issue 2: Concatenation of tensors with different data types when return_indices=True.
def test_return_indices_concat_dtype_error():
    # Create a valid float32 input tensor.
    x = torch.randn(2, 3, 16, device="cuda", dtype=torch.float32)
    # Use kernel parameters that produce a valid output.
    kernel_size = 3
    stride = 1
    padding = 1
    dilation = 1
    module = build_kernel()
    # Expect the concatenation to fail because output is float and indices is int64.
    with pytest.raises(RuntimeError):
        # This should trigger an error on torch::cat in the C++ forward.
        module.forward(x, kernel_size, stride, padding, dilation, True)

# Issue 3: No CUDA error checking may hide configuration issues.
# We simulate this by configuring the pooling parameters so that the computed output_length is non-positive.
def test_invalid_output_length():
    # Create a valid float32 input tensor.
    # We choose kernel parameters that yield a negative or zero output_length
    # Using a very high dilation on a small input will cause output_length <= 0.
    x = torch.randn(2, 3, 8, device="cuda", dtype=torch.float32)
    kernel_size = 3
    stride = 1
    padding = 0
    dilation = 10  # This should yield a negative output length.
    module = build_kernel()
    
    with pytest.raises(RuntimeError):
        # The TORCH_CHECK in the forward C++ code should trigger.
        module.forward(x, kernel_size, stride, padding, dilation, False)
