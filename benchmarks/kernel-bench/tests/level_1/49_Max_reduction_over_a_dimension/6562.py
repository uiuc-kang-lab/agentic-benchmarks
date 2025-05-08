
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Helper function to compute reduction over a given dim using PyTorch
def torch_max_reduce(input, dim):
    return torch.max(input, dim=dim)[0]

# Test case 1: Trigger shared memory type mismatch by using half precision.
def test_half_precision_mismatch():
    # Create an input tensor with half precision.
    batch_size, dim1, dim2 = 8, 32, 16
    # Ensure the reduction dimension is the middle dim (dim1)
    input_tensor = torch.randn(batch_size, dim1, dim2, device="cuda", dtype=torch.half)
    
    # Build the CUDA kernel module.
    my_module = build_kernel()
    
    # Call our CUDA kernel with reduction over dimension 1.
    # Our kernel expects an int64_t for the dimension.
    output_cuda = my_module.forward(input_tensor, 1)
    
    # Compute reference using PyTorch
    output_ref = torch_max_reduce(input_tensor, dim=1)
    
    # Because of the shared memory size mis-allocation (using sizeof(float) instead of sizeof(half)),
    # the kernel might produce incorrect results.
    assert not torch.allclose(output_cuda, output_ref), \
        "The kernel appears to work correctly on half precision even though shared memory allocation is wrong."

# Test case 2: Trigger max function issues by using double precision.
def test_double_precision_max_issue():
    # Create an input tensor with double precision.
    batch_size, dim1, dim2 = 8, 32, 16
    # Reduce along dim2
    input_tensor = torch.randn(batch_size, dim1, dim2, device="cuda", dtype=torch.double)
    
    my_module = build_kernel()
    
    # Call CUDA kernel with reduction over dimension 2.
    output_cuda = my_module.forward(input_tensor, 2)
    
    # Compute the expected result with PyTorch.
    output_ref = torch_max_reduce(input_tensor, dim=2)
    
    # Due to the unqualified use of max() in the kernel, the double precision version may be handled incorrectly.
    # We expect a noticeable discrepancy.
    assert not torch.allclose(output_cuda, output_ref, atol=1e-6), \
        "The kernel produced correct results on double precision despite potential max() function issues."

# Test case 3: Check for inefficiency in the kernel (this is non-functional but can serve as a warning trigger).
def test_unnecessary_syncs():
    # While we cannot measure performance regression reliably in a unit test, we can at least run a case with
    # large reduction dimension so that many redundant synchronizations occur.
    batch_size, dim1, dim2 = 4, 1024, 32
    input_tensor = torch.randn(batch_size, dim1, dim2, device="cuda", dtype=torch.float32)
    
    my_module = build_kernel()
    output_cuda = my_module.forward(input_tensor, 1)
    output_ref = torch_max_reduce(input_tensor, dim=1)
    
    # In a correct implementation the results should match. However, due to the unnecessary use of __syncthreads(),
    # this test will primarily serve as a baseline for performance observation rather than correctness.
    assert torch.allclose(output_cuda, output_ref, atol=1e-5), \
        "Kernel output differs from reference even though only performance (inefficiency) is at stake."

if __name__ == "__main__":
    pytest.main([__file__])
