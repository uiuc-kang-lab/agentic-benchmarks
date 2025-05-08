
import pytest
import torch
from torch.utils.cpp_extension import load
import os

def build_kernel(extra_cuda_cflags=None):
    src_file = os.path.join(os.path.dirname(__file__), "kernel.cu")
    cuda_module = load(
        name="cuda_tanh_module",
        sources=[src_file],
        extra_cuda_cflags=extra_cuda_cflags if extra_cuda_cflags is not None else ["-O3", "--use_fast_math"],
        verbose=True,
    )
    return cuda_module

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_double_precision():
    """
    Test issue 1:
    The kernel calls tanhf even when the tensor is of double type.
    We create a double tensor input, run the kernel, and compare with torch.tanh output.
    They are expected to differ beyond acceptable tolerance because of precision issues.
    """
    # Create a double precision tensor on CUDA
    input_tensor = torch.randn(1024, device="cuda", dtype=torch.double)
    # Build module and invoke the kernel
    cuda_module = build_kernel()
    # Call the kernel; note: our forward function expects input to be a CUDA tensor.
    output = cuda_module.forward(input_tensor)
    # Compute reference output with torch.tanh (which uses double precision math)
    output_ref = torch.tanh(input_tensor)
    
    # The results are expected to be different due to using tanhf in kernel.
    # We assert that they are not within a tight tolerance.
    diff = (output.double() - output_ref).abs().max().item()
    assert diff > 1e-4, f"Double precision test: the kernel output matches torch.tanh unexpectedly; diff = {diff}"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_non_contiguous_input():
    """
    Test issue 3:
    The kernel assumes the input tensor is contiguous.
    We create a non-contiguous tensor by transposing a 2D tensor and reshaping.
    This test checks if the kernel produces an incorrect result or crashes.
    """
    # Create a contiguous tensor and then make it non-contiguous by transposing.
    x = torch.randn(64, 64, device="cuda", dtype=torch.float32)
    x_noncontig = x.t()  # Transpose makes it non-contiguous in memory.
    
    # Although PyTorch operations like torch.tanh work on non-contiguous tensors,
    # our kernel directly uses data_ptr(), so the result may be erroneous.
    cuda_module = build_kernel()
    
    # Catch potential errors from non-contiguous memory access.
    try:
        output = cuda_module.forward(x_noncontig)
        # Compare with reference (we force input to be contiguous for the reference)
        output_ref = torch.tanh(x_noncontig.contiguous()).view_as(x_noncontig)
        # It is likely that the output will not match.
        if torch.allclose(output, output_ref, atol=1e-5):
            pytest.fail("Kernel unexpectedly handled non-contiguous input correctly, but it was assumed not to.")
    except Exception as e:
        # In some cases, a crash or exception might be what we see.
        pytest.skip("Kernel failed due to non-contiguous input as expected: " + str(e))

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_custom_block_configuration():
    """
    Test issue 2:
    The kernel uses a fixed TILE_WIDTH shared memory array and assumes blockDim.x == TILE_WIDTH.
    We simulate a condition where this assumption is violated. Although the provided forward()
    always launches with TILE_WIDTH threads, we recompile the kernel with a forced alternative tile size.
    
    Note: This is a synthetic test to highlight potential misuse if someone modifies the kernel launch config.
    """
    # Here we force a different block configuration via a compiler definition.
    # For example, we define TILE_WIDTH to be 16 at compile time using extra_cuda_cflags.
    extra_flags = ["-O3", "--use_fast_math", "-DTILE_WIDTH=16"]
    cuda_module = build_kernel(extra_cuda_cflags=extra_flags)
    # Create an input tensor of size that is not a multiple of the new tile width.
    input_tensor = torch.randn(1024 + 7, device="cuda", dtype=torch.float32)
    
    # Call the forward; since the kernel code still allocates shared memory of size TILE_WIDTH (as defined in code)
    # but our launch configuration in forward does not change, potential mis-indexing may happen.
    # In this test, we expect the kernel output to be incorrect.
    output = cuda_module.forward(input_tensor)
    output_ref = torch.tanh(input_tensor)
    if torch.allclose(output, output_ref, atol=1e-5):
        pytest.fail("Kernel did not exhibit expected error with mismatched block configuration and shared memory tile size.")
