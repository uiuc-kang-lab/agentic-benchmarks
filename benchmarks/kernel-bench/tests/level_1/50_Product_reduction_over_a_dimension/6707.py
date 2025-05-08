
import pytest
import torch
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

# Issue 1: Incorrect indexing for general multidimensional reduction.
# For a contiguous tensor that is 3D and reducing along a middle dimension,
# the expected product computed by torch.prod will not match the kernel's output.
def test_incorrect_indexing():
    my_module = build_kernel()
    # Create a 3D tensor which is contiguous.
    batch_size = 4
    dim1 = 8
    dim2 = 10
    # reduction over dimension 1 (middle dimension).
    x = torch.randn(batch_size, dim1, dim2, device="cuda", dtype=torch.float32)
    
    # Expected output computed via PyTorch, product over dimension 1.
    ref = torch.prod(x, dim=1)
    
    # Kernel forward expects input tensor and reduction dimension.
    # The kernel’s forward implementation computes output = optimized_prod_kernel(x, dim)
    # but its indexing is flawed when reduction dim is not the first or last.
    out = my_module.forward(x, 1)
    
    # They should be close if the kernel was correct.
    # Here we expect a mismatch due to incorrect indexing.
    assert not torch.allclose(out, ref, atol=1e-5), (
        f"Test failed: Kernel produced output matching torch.prod, but it should have an indexing error."
    )

# Issue 2: Lack of support for data types other than float32.
# Passing in a double tensor will cause the kernel to reinterpret memory (or even crash)
# because it always treats the data as 32-bit floats.
def test_incorrect_dtype():
    my_module = build_kernel()
    # Create a contiguous tensor of type double.
    batch_size = 4
    dim0 = 10
    dim1 = 12
    # Using reduction on the first dimension.
    x = torch.randn(batch_size, dim0, device="cuda", dtype=torch.float64)
    
    # Even though torch.prod can handle double tensors, our kernel will use x.data_ptr<float>()
    # Hence, the result from the kernel will likely be numerically different from the expected result.
    ref = torch.prod(x, dim=0)
    
    out = my_module.forward(x, 0)
    
    # They will be very different because the kernel treated the data as float.
    assert not torch.allclose(out, ref, atol=1e-5), (
        "Test failed: Kernel output for double tensor matches reference output, but it should be incorrect."
    )

# Issue 3: The kernel requires the input tensor to be contiguous.
# Passing in a noncontiguous tensor should trigger a CHECK_INPUT failure.
def test_noncontiguous_input():
    my_module = build_kernel()
    # Create a tensor and then make it noncontiguous by permuting dimensions.
    x = torch.randn(4, 8, 10, device="cuda", dtype=torch.float32).permute(1, 0, 2)
    # Even though the resulting tensor may be contiguous in memory sometimes, permute usually yields noncontiguity.
    if x.is_contiguous():
        # Force noncontiguity by taking a slice with a step.
        x = x[::, ::2, ::]
    assert not x.is_contiguous(), "Test setup error: x should be noncontiguous."
    
    # The CHECK_INPUT macro in the kernel should raise an error.
    with pytest.raises(RuntimeError, match="must be contiguous"):
        _ = my_module.forward(x, 1)
