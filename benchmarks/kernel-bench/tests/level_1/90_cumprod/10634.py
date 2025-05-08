
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the CUDA module from kernel.cu
    cuda_module = load(
        name="cumprod_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

def test_incorrect_offset_calculation():
    """
    Trigger Issue 1:
    Use a higher-dimensional tensor where the cumulative product dimension is not the last dimension.
    The kernel’s simplistic offset calculation will produce a wrong result.
    """
    my_module = build_kernel()
    # Create a tensor with shape (2, 3, 4) and perform cumprod along dim=1.
    input_tensor = torch.randn(2, 3, 4, device="cuda", dtype=torch.float32)
    # Our kernel is expected to run cumprod along dim 1.
    # The PyTorch built-in function gives the correct result.
    expected = torch.cumprod(input_tensor, dim=1)
    # Call the kernel: note that the kernel forward takes (input, dim)
    output = my_module.forward(input_tensor, 1)
    # Check that the outputs differ because the offset calculation is wrong.
    # We trigger the issue by asserting that the output is NOT equal to the expected result.
    if torch.allclose(output, expected, atol=1e-5):
        pytest.fail("Kernel offset calculation appears to be correct for a non-2D tensor, "
                    "but an error was expected due to incorrect offset computation.")

def test_non_contiguous_input():
    """
    Trigger Issue 2:
    Pass a non-contiguous tensor through the kernel.
    The kernel does not handle non-contiguous memory correctly.
    """
    my_module = build_kernel()
    # Create a contiguous tensor and then make it non-contiguous via transpose.
    input_tensor = torch.randn(10, 20, device="cuda", dtype=torch.float32)
    non_contig = input_tensor.transpose(0, 1)  # Now non-contiguous
    # Choose a dimension that corresponds to one of the dimensions in non_contig.
    # For transpose(0,1), the memory layout is swapped and our kernel's offset math will be wrong.
    expected = torch.cumprod(non_contig, dim=0)
    output = my_module.forward(non_contig, 0)
    if torch.allclose(output, expected, atol=1e-5):
        pytest.fail("Kernel handled a non-contiguous input correctly, but it was expected to fail.")

def test_invalid_dimension():
    """
    Trigger Issue 3:
    Pass an invalid dimension index to the kernel.
    The kernel doesn’t validate the dimension argument and may access out-of-range memory.
    """
    my_module = build_kernel()
    input_tensor = torch.randn(5, 10, device="cuda", dtype=torch.float32)
    # Pass an invalid dimension (e.g., 2 when tensor has only dimensions 0 and 1).
    with pytest.raises(RuntimeError):
        # Expecting a CUDA error or runtime exception due to an invalid dimension.
        _ = my_module.forward(input_tensor, 2)

def test_kernel_launch_error_checking():
    """
    Trigger Issue 4:
    This test simulates a scenario where incorrect kernel launch parameters might lead to an error.
    Since the kernel does not check for errors after launch, we force an error situation.
    One way is to pass an empty tensor, which might lead to zero total_threads.
    """
    my_module = build_kernel()
    # Create an empty tensor input.
    input_tensor = torch.empty(0, device="cuda", dtype=torch.float32)
    # Default behavior of torch.cumprod on empty tensors is to return an empty tensor.
    # We call the kernel and see if any error is raised (expecting that the lack of error checking might hide issues).
    try:
        output = my_module.forward(input_tensor, 0)
        # If no exception, check if output is empty as expected.
        if output.numel() != 0:
            pytest.fail("Kernel output is not empty for an empty input tensor, indicating potential launch issues.")
    except Exception as e:
        # If an exception is thrown, the test is successful in exposing a lack of proper error handling.
        pass
