
import pytest
import torch
from torch.utils.cpp_extension import load

# Build and load the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="conv_transpose2d_ext",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Kernel only supports float32.
# Create inputs with half precision and expect a runtime error (or wrong computation)
def test_dtype_not_float32():
    cuda_module = build_kernel()
    batch_size = 4
    in_channels = 3
    height_in = 8
    width_in = 8
    out_channels = 2
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1

    # Create tensors in half precision
    input_tensor = torch.randn(batch_size, in_channels, height_in, width_in, device="cuda", dtype=torch.half)
    # Weight shape for ConvTranspose2d is (in_channels, out_channels, kernel_size, kernel_size)
    weight_tensor = torch.randn(in_channels, out_channels, kernel_size, kernel_size, device="cuda", dtype=torch.half)
    bias_tensor = torch.randn(out_channels, device="cuda", dtype=torch.half)

    with pytest.raises(RuntimeError):
        # Expect that the kernel (which uses data_ptr<float> unconditionally)
        # will error out when passed half precision tensors.
        cuda_module.forward(input_tensor, weight_tensor, bias_tensor, stride, padding, dilation)

# Issue 2: Kernel assumes a square kernel.
# Create a weight tensor with non-square shape and check that the output does not match the reference.
def test_non_square_kernel():
    cuda_module = build_kernel()
    batch_size = 2
    in_channels = 3
    height_in = 10
    width_in = 10
    out_channels = 4
    # Create a non-square kernel, for example (3, 4) instead of (3, 3)
    kernel_height = 3
    kernel_width = 4  # asymmetric dimension; kernel.cu will use kernel_height for both dimensions.
    stride = 1
    padding = 1
    dilation = 1

    # Input is properly float32.
    input_tensor = torch.randn(batch_size, in_channels, height_in, width_in, device="cuda", dtype=torch.float32)
    # Create weight with non-square shape (but our kernel will use weight.size(2) for both dims)
    weight_tensor = torch.randn(in_channels, out_channels, kernel_height, kernel_width, device="cuda", dtype=torch.float32)
    bias_tensor = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    # Get output from our custom CUDA kernel.
    out_custom = cuda_module.forward(input_tensor, weight_tensor, bias_tensor, stride, padding, dilation)

    # Calculate reference output using torch.nn.functional.conv_transpose2d, but note:
    # PyTorch expects the kernel to be of shape (in_channels, out_channels, kernel_height, kernel_width)
    # Our kernel erroneously uses kernel_height for both spatial dims.
    out_reference = torch.nn.functional.conv_transpose2d(input_tensor, weight_tensor, bias=bias_tensor,
                                                         stride=stride, padding=padding, dilation=dilation)
    # The results should differ because of the incorrect assumption of a square kernel.
    # We use a loose tolerance to assert that the error is nonzero.
    assert not torch.allclose(out_custom, out_reference, atol=1e-4), (
        "The custom kernel output unexpectedly matches the reference output with a non-square kernel."
    )

# Issue 3: Kernel assumes blockDim.x is a multiple of 32.
# Although the launcher in the kernel always uses blockDim.x=128, a more general usage scenario
# might not satisfy this assumption. We simulate a scenario that stresses the warp-level reduction.
def test_blockdim_multiple_of_32_assumption():
    cuda_module = build_kernel()
    # Use small tensor sizes such that the total number of output elements is very low.
    # This creates a situation where very few warps are active and some warps may not be fully utilized.
    batch_size = 1
    in_channels = 2
    height_in = 4
    width_in = 4
    out_channels = 1
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1

    input_tensor = torch.randn(batch_size, in_channels, height_in, width_in, device="cuda", dtype=torch.float32)
    weight_tensor = torch.randn(in_channels, out_channels, kernel_size, kernel_size, device="cuda", dtype=torch.float32)
    bias_tensor = torch.randn(out_channels, device="cuda", dtype=torch.float32)

    # Compute the output from our custom kernel.
    out_custom = cuda_module.forward(input_tensor, weight_tensor, bias_tensor, stride, padding, dilation)

    # Compute reference output from PyTorch's conv_transpose2d.
    out_reference = torch.nn.functional.conv_transpose2d(input_tensor, weight_tensor, bias=bias_tensor,
                                                         stride=stride, padding=padding, dilation=dilation)
    # Even for small outputs the kernel should compute correct results.
    # We expect a discrepancy if the warp-level reduction assumption is faulty.
    assert not torch.allclose(out_custom, out_reference, atol=1e-4), (
        "The custom kernel output unexpectedly matches the reference output with a non-standard blockDim usage."
    )
