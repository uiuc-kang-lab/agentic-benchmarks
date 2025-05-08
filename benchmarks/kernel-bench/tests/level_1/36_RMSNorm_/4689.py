
import pytest
import torch
from torch.utils.cpp_extension import load

# Helper function to build the CUDA extension module from kernel.cu
def build_kernel():
    cuda_module = load(
        name="rms_norm_kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# A helper function to compute RMS normalization in PyTorch on the CPU.
def pytorch_rms_norm(x, eps):
    # x: [batch, features, ...] (any number of trailing dims)
    # Compute RMS along dim=1 (the features) and broadcast it.
    rms = torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + eps)
    return x / rms

# Issue 1: eps type mismatch.
# Create a double tensor with very small values so the eps value matters.
def test_double_eps_issue():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
        
    # Use double precision input.
    batch_size = 4
    num_features = 32
    dim1, dim2 = 8, 8
    eps = 1e-5  # eps passed as float from python
    # Use values close to zero so that rounding differences become visible.
    x = (torch.randn(batch_size, num_features, dim1, dim2, device="cuda", dtype=torch.float64) * 1e-3)
    
    kernel = build_kernel()
    # Call our CUDA kernel function
    out_cuda = kernel.forward(x, eps)
    # Compute reference output with PyTorch CPU version (move to cuda later)
    out_ref = pytorch_rms_norm(x, eps)
    
    # The error might be slight, but for double precision we expect high accuracy.
    torch.cuda.synchronize()
    # Use a very strict tolerance since any eps conversion issue would show up.
    assert torch.allclose(out_cuda, out_ref, atol=1e-12), "Double-precision eps conversion issue detected!"

# Issue 2: Non-contiguous input tensor.
# Pass an input tensor which has been transposed so that its memory is noncontiguous.
def test_non_contiguous_input():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
        
    batch_size = 4
    num_features = 16
    dim1, dim2 = 16, 16
    eps = 1e-5
    
    # Create a contiguous tensor and then perform a transpose on a subset of dimensions.
    x = torch.randn(batch_size, num_features, dim1, dim2, device="cuda", dtype=torch.float32)
    # Transpose the last two dimensions to force non-contiguity.
    x_noncontig = x.transpose(2, 3)
    # Although logically the same data, the kernel indexing assumes a contiguous layout.
    
    kernel = build_kernel()
    out_cuda = kernel.forward(x_noncontig, eps)
    out_ref = pytorch_rms_norm(x_noncontig, eps)
    
    torch.cuda.synchronize()
    # We expect the results to differ because the kernel’s indexing is incorrect on noncontiguous layouts.
    assert not torch.allclose(out_cuda, out_ref, atol=1e-5), "Kernel incorrectly handled a non-contiguous input tensor!"

# Issue 3: Loop unrolling with unexpected num_features.
# Use a number of features that is not a multiple of 4 to potentially expose unroll-related issues.
def test_unroll_issue():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
        
    batch_size = 4
    num_features = 7  # Not a multiple of 4
    dim1, dim2 = 16, 16
    eps = 1e-5
    x = torch.randn(batch_size, num_features, dim1, dim2, device="cuda", dtype=torch.float32)
    
    kernel = build_kernel()
    out_cuda = kernel.forward(x, eps)
    out_ref = pytorch_rms_norm(x, eps)
    
    torch.cuda.synchronize()
    # We expect a disagreement because the unrolled loop may not correctly handle cases where
    # num_features does not align with the unroll factor.
    assert not torch.allclose(out_cuda, out_ref, atol=1e-5), ("Kernel output for non-multiple-of-4 features "
                                                             "should not match reference if unrolling is miscompiled!")

# Issue 4: Lack of error checking post kernel launch.
# Pass an unsupported input type (e.g. integer type) to force an error from the kernel.
def test_input_type_error():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
        
    batch_size = 4
    num_features = 16
    dim1, dim2 = 16, 16
    eps = 1e-5
    # Create an integer tensor; our kernel supports only floating point types.
    x_int = torch.randint(0, 10, (batch_size, num_features, dim1, dim2), device="cuda", dtype=torch.int32)
    
    kernel = build_kernel()
    with pytest.raises(RuntimeError):
        # The CUDA kernel dispatch macro should not instantiate the kernel for int32, so an error is expected.
        _ = kernel.forward(x_int, eps)
