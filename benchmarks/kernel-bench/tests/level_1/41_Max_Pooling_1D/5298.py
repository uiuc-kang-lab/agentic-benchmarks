
import torch
import pytest
from torch.utils.cpp_extension import load

# Builds the CUDA extension from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="maxpool1d_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Test that when return_indices is True the returned tensor 
# contains both pooling output and indices concatenated along the last dimension,
# violating the expected API (which should return a tuple).
def test_return_indices_format():
    # Create a simple input tensor.
    batch_size = 2
    num_channels = 3
    seq_length = 10
    kernel_size = 2
    stride = 1
    padding = 0
    dilation = 1
    return_indices = True

    x = torch.randn(batch_size, num_channels, seq_length, device="cuda", dtype=torch.float32)

    module = build_kernel()
    # Call the forward kernel function.
    # Note: The kernel's forward function returns a single tensor that is a concatenation of output and indices.
    result = module.forward(x, kernel_size, stride, padding, dilation, return_indices)
    torch.cuda.synchronize()

    # Calculate expected output_length using the formula: ((L+2p-d*(k-1)-1)/stride)+1.
    output_length = ((seq_length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1

    # The expected behavior (as per PyTorch API) would be a tuple (output, indices) each of shape:
    # [batch_size, num_channels, output_length]. Instead, the kernel returns a single tensor with shape:
    # [batch_size, num_channels, 2 * output_length]
    expected_shape = (batch_size, num_channels, 2 * output_length)
    assert result.shape == expected_shape, (
        f"Expected concatenated tensor shape {expected_shape} when return_indices is True, "
        f"but got {result.shape}. This indicates an output format issue."
    )

# Issue 2: Test that the kernel does not support non-float32 data types.
def test_dtype_not_float():
    # Create an input tensor in double precision.
    batch_size = 2
    num_channels = 3
    seq_length = 10
    kernel_size = 2
    stride = 1
    padding = 0
    dilation = 1
    return_indices = False

    x = torch.randn(batch_size, num_channels, seq_length, device="cuda", dtype=torch.float64)

    module = build_kernel()

    with pytest.raises(RuntimeError) as excinfo:
        # The kernel is written to use float and should raise an error if given a double tensor.
        module.forward(x, kernel_size, stride, padding, dilation, return_indices)
        torch.cuda.synchronize()
    err_msg = str(excinfo.value)
    # Check for a message indicating that the input must be of type float.
    assert "float" in err_msg.lower(), (
        f"Expected a type error related to float32 datatype but got: {err_msg}"
    )

# Issue 3: Test that lack of kernel launch error-checking leads to silent failures.
# We trigger a situation where the kernel launch should fail.
def test_kernel_launch_error():
    # Here we use a non-contiguous input tensor. The forward function explicitly checks for contiguity,
    # and thus an error is expected.
    batch_size = 2
    num_channels = 3
    seq_length = 10
    kernel_size = 2
    stride = 1
    padding = 0
    dilation = 1
    return_indices = False

    x = torch.randn(batch_size, num_channels, seq_length, device="cuda", dtype=torch.float32)
    # Make the input non-contiguous by transposing two dimensions.
    x_non_contig = x.transpose(1, 2)

    module = build_kernel()
    with pytest.raises(RuntimeError) as excinfo:
        module.forward(x_non_contig, kernel_size, stride, padding, dilation, return_indices)
        torch.cuda.synchronize()
    err_msg = str(excinfo.value)
    assert "contiguous" in err_msg, (
        f"Expected error regarding non-contiguous input tensor, but got: {err_msg}"
    )
