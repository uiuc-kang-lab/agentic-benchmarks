
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="l2_norm_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test Case 1:
# This test is intended to expose the issue in handling vectorized loads/stores for double precision.
# For a double tensor with a channel size that does not meet the expected alignment for float (e.g. C=5),
# the kernel will process the first two elements vectorized and then the remaining elements in the scalar loop,
# skipping elements in the middle.
def test_vectorized_double_alignment():
    my_module = build_kernel()
    batch_size = 4
    C = 5  # Deliberately chosen so that (C/4)*4 is 4, which is wrong for double (should use multiples of 2).
    # Create a double tensor (dtype=torch.double) with contiguous layout.
    x = torch.randn(batch_size, C, dtype=torch.double, device='cuda')
    # Compute the expected output using PyTorch native operations.
    norm = torch.norm(x, p=2, dim=1, keepdim=True) + 1e-12
    expected = x / norm

    # Launch the kernel function.
    output = my_module.forward(x)
    torch.cuda.synchronize()

    # The output from the kernel is expected to be different from expected due to alignment issue.
    # We check that they do not match (i.e. the bug is triggered).
    assert not torch.allclose(output, expected, atol=1e-5), \
        "Test did not trigger the double vectorization misalignment bug as expected."

# Test Case 2:
# This test targets the issue with assumptions on the normalization dimension and tensor layout.
# If a tensor is constructed with a non-standard memory layout (e.g. by transposing a contiguous tensor),
# the kernel’s computation (using outer_stride = input.stride(0) and stride_C = input.stride(1))
# may result in wrong indexing and an incorrect normalization result.
def test_non_contiguous_input():
    my_module = build_kernel()
    batch_size = 16
    C = 16384
    # Create a 2D tensor and then transpose it so that normalization (supposed to occur along dim=1)
    # is no longer along a contiguous dimension.
    x = torch.randn(batch_size, C, device='cuda', dtype=torch.float32)
    x_t = x.t()  # Transpose to break the assumption on dims (now shape is [C, batch_size]).
    
    # For comparison we perform normalization along the original dim after transposing back.
    # The expected result is computed from the original tensor.
    norm = torch.norm(x, p=2, dim=1, keepdim=True) + 1e-12
    expected = x / norm
    
    # Now call the kernel on the non-contiguous tensor.
    # The kernel assumes normalization along dim=1 in its indexing,
    # so if the tensor is non-contiguous in that dimension the result will be wrong.
    output_t = my_module.forward(x_t)
    # Transpose the result back to match the original shape.
    output = output_t.t()
    torch.cuda.synchronize()
    
    # The kernel output should differ from the expected output because the indexing assumptions are violated.
    assert not torch.allclose(output, expected, atol=1e-5), \
        "Test did not trigger the non-contiguous input bug as expected."
