
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build the CUDA extension from kernel.cu
def build_kernel():
    # Ensure that the file kernel.cu is in the same directory as this test file.
    module = load(
        name="rms_norm_module",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return module

# A helper function to compute RMS normalization in PyTorch
def torch_rms_norm(x: torch.Tensor, eps: float):
    # Compute RMS along feature dimension (dim=1) while keeping dimensions
    rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + eps)
    return x / rms

# Test case 1: Non-contiguous input tensor trigger.
def test_non_contiguous_input():
    module = build_kernel()
    batch_size = 4
    num_features = 64
    H, W = 8, 8
    eps = 1e-5
    # Create a contiguous input and then make it non-contiguous by transposing one of the spatial dimensions.
    x = torch.randn(batch_size, num_features, H, W, device="cuda", dtype=torch.float32)
    x_non_contig = x.permute(0, 1, 3, 2)  # Now non-contiguous; shape still (batch, features, W, H)
    
    # Use the kernel; since the kernel assumes a contiguous layout with
    # numel_per_batch computed as product over dimensions from dim=2,
    # the non-contiguous (and transposed) input may yield incorrect results.
    output_cuda = module.forward(x_non_contig, eps)
    # Compute reference using PyTorch RMS norm (which works with non-contiguous tensors)
    output_ref = torch_rms_norm(x_non_contig, eps)
    
    # They will differ if the kernel does not handle non-contiguity.
    with pytest.raises(AssertionError):
        # Expectation: the outputs should be equal. If they aren’t,
        # it indicates the kernel assumption on contiguity is problematic.
        assert torch.allclose(output_cuda, output_ref, atol=1e-4)

# Test case 2: Double precision tensor trigger.
def test_double_precision_input():
    module = build_kernel()
    batch_size = 4
    num_features = 64
    H, W = 8, 8
    eps = 1e-5
    # Create a double precision tensor on CUDA.
    x_double = torch.randn(batch_size, num_features, H, W, device="cuda", dtype=torch.float64)
    # The kernel dispatch macros do not support double (only float and half).
    # Thus, we expect an error.
    with pytest.raises(RuntimeError):
        module.forward(x_double, eps)

# Test case 3: Feature dimension not a multiple of 32.
def test_feature_dim_non_multiple_of_32():
    module = build_kernel()
    batch_size = 4
    # Choose num_features that is not divisible by 32.
    num_features = 45  
    H, W = 8, 8
    eps = 1e-5
    x = torch.randn(batch_size, num_features, H, W, device="cuda", dtype=torch.float32)
    
    output_cuda = module.forward(x, eps)
    output_ref = torch_rms_norm(x, eps)
    
    # The kernel should still compute valid results even if the unrolled loop is not perfect.
    assert torch.allclose(output_cuda, output_ref, atol=1e-4), \
        f"Output mismatches for non-multiple-of-32 features. Diff: {(output_cuda - output_ref).abs().max()}"

# Test case 4: Check for potential precision issues due to 0.0f initialization
def test_mixed_precision_issue():
    module = build_kernel()
    batch_size = 4
    num_features = 64
    H, W = 8, 8
    eps = 1e-5
    # Even though the dispatch does not include double, we test with half to see if precison is maintained.
    x_half = torch.randn(batch_size, num_features, H, W, device="cuda", dtype=torch.half)
    
    output_cuda = module.forward(x_half, eps)
    output_ref = torch_rms_norm(x_half.float(), eps).half()
    
    assert torch.allclose(output_cuda, output_ref, atol=1e-2), \
        f"Output mismatches for half precision input. Diff: {(output_cuda - output_ref).abs().max()}"

if __name__ == "__main__":
    pytest.main([__file__])
