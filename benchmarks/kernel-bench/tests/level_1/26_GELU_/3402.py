
import torch
import pytest
from torch.utils.cpp_extension import load
import os

# Utility function to build and load the CUDA kernel module
def build_kernel():
    # Ensure that we compile using the current directory's kernel.cu file
    this_dir = os.path.dirname(os.path.abspath(__file__))
    cuda_module = load(
        name="test_module",
        sources=[os.path.join(this_dir, "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Verify that kernel errors out when a non-contiguous tensor is provided.
def test_non_contiguous_tensor():
    cuda_module = build_kernel()
    # Create a contiguous tensor and then a non-contiguous view (via transpose)
    x = torch.randn(16, 16384, device="cuda", dtype=torch.float32)
    x_non_contiguous = x.t()  # Transpose makes it non-contiguous
    with pytest.raises(RuntimeError) as excinfo:
        # The kernel assumes contiguous memory layout, might produce an error or incorrect result.
        cuda_module.forward(x_non_contiguous)
    assert "contiguous" in str(excinfo.value).lower() or "layout" in str(excinfo.value).lower()

# Test 2: Verify that kernel errors out when the input tensor is not float32.
def test_wrong_dtype():
    cuda_module = build_kernel()
    # Create a tensor with dtype float64
    x = torch.randn(16, 16384, device="cuda", dtype=torch.float64)
    with pytest.raises(RuntimeError) as excinfo:
        cuda_module.forward(x)
    assert "float32" in str(excinfo.value)

# Test 3: Verify kernel behavior with input sizes that are not a multiple of 256.
def test_input_size_not_multiple_of_block_dim():
    cuda_module = build_kernel()
    # Create a contiguous tensor with number of elements that does not divide evenly into 256.
    # For example, shape (17, 1000) => 17000 elements, and 17000 % 256 != 0.
    x = torch.randn(17, 1000, device="cuda", dtype=torch.float32)
    # Although the kernel can process out-of-bound threads using the if conditions,
    # using shared memory in this way might lead to unexpected inefficiencies.
    # Here we verify the correctness of the computation as a sanity check.
    y_cuda = cuda_module.forward(x)
    # Using the CPU version of gelu from torch.nn.functional for reference
    y_ref = torch.nn.functional.gelu(x)
    torch.cuda.synchronize()
    # We use a relatively loose tolerance given potential differences due to fused operations
    assert torch.allclose(y_cuda, y_ref, atol=1e-5), "Kernel computation failed on non-multiple of blockDim input."
    
# Test 4: Verify that the kernel handles very small inputs (testing the hard-coded block size and shared memory allocation).
def test_small_input():
    cuda_module = build_kernel()
    # Create a very small tensor, smaller than the block size (e.g., only 10 elements)
    x = torch.randn(10, device="cuda", dtype=torch.float32)
    y_cuda = cuda_module.forward(x)
    y_ref = torch.nn.functional.gelu(x)
    torch.cuda.synchronize()
    assert torch.allclose(y_cuda, y_ref, atol=1e-5), "Kernel computation failed on small input."

# Test 5: Check for asynchronous kernel errors by forcing a device synchronization after kernel execution.
def test_async_error_detection():
    cuda_module = build_kernel()
    # Intentionally create an input tensor that is non-contiguous to check for asynchronous error detection.
    x = torch.randn(16, 16384, device="cuda", dtype=torch.float32)
    x_non_contiguous = x.t()  # now non-contiguous
    # Call forward() without immediate error check and then force synchronization.
    with pytest.raises(RuntimeError) as excinfo:
        output = cuda_module.forward(x_non_contiguous)
        # Force synchronization to catch asynchronous errors
        torch.cuda.synchronize()
    assert "contiguous" in str(excinfo.value).lower() or "layout" in str(excinfo.value).lower()
