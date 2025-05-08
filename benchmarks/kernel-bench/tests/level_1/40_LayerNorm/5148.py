
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="layernorm_extension",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Reference implementation using PyTorch's built-in LayerNorm
def reference_layernorm(x, weight, bias, eps):
    # We assume the normalization is over the last len(weight.shape) dims.
    normalized_shape = weight.shape
    return torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)

# Test 1: Noncontiguous Input
# This test creates a noncontiguous tensor by transposing
# and verifies that the CUDA kernel output deviates from the expected value.
def test_noncontiguous_input():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    torch.cuda.synchronize()
    batch = 16
    # Let's use a normalized shape of (8, 16) so that weight and bias have 128 elements.
    normalized_shape = (8, 16)
    # Create an input of shape (batch, *normalized_shape)
    x = torch.randn(batch, *normalized_shape, device='cuda')
    # Make the tensor noncontiguous by transposing two dimensions before sending to kernel.
    x = x.transpose(0, 1)  # Now noncontiguous and the trailing dimensions are not the same!
    # Create weight and bias (assume they are contiguous)
    weight = torch.randn(*normalized_shape, device='cuda')
    bias = torch.randn(*normalized_shape, device='cuda')
    
    my_module = build_kernel()
    # The kernel expects the normalized size to match weight.numel() and uses pointer arithmetic.
    # Because x is noncontiguous the kernel will compute wrong values.
    output = my_module.forward(x, weight, bias)
    ref = reference_layernorm(x.contiguous(), weight, bias, 1e-5)  # forcing contiguous for reference
    torch.cuda.synchronize()
    
    # We expect a significant difference because noncontiguous memory causes index errors.
    diff = (output - ref).abs().max()
    assert diff > 1e-3, f"Noncontiguous input did not trigger the expected error. Max diff: {diff}"

# Test 2: Mismatched trailing dimensions (normalized shape mismatch)
# The kernel computes outer_size = x.numel()/weight.numel(). If x's trailing dimensions do not match weight.numel(),
# the kernel index arithmetic goes awry. We simulate this by providing an input where x.numel() is not divisible by weight.numel().
def test_mismatched_normalized_shape():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    torch.cuda.synchronize()
    batch = 16
    # Expected normalized shape is (10,) so normalized_size=10.
    normalized_shape = (10,)
    # Create an input tensor whose last dimension is different from 10 (say 9) so that the division is wrong.
    x = torch.randn(batch, 9, device='cuda')
    weight = torch.randn(*normalized_shape, device='cuda')
    bias = torch.randn(*normalized_shape, device='cuda')
    
    my_module = build_kernel()
    with pytest.raises(RuntimeError):
        # The kernel’s indexing will go out-of-bounds and should trigger an error.
        _ = my_module.forward(x, weight, bias)

# Test 3: Normalized size smaller than warp size
# The kernel uses a fixed full-warp mask (0xFFFFFFFF) in __shfl_down_sync.
# When normalized_size is less than 32, the fixed mask may cause incorrect results.
def test_small_normalized_size():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    torch.cuda.synchronize()
    batch = 32
    # normalized_shape with few elements so that normalized_size < 32.
    normalized_shape = (8,)  # normalized_size=8, which is less than a warp.
    x = torch.randn(batch, *normalized_shape, device='cuda')
    weight = torch.randn(*normalized_shape, device='cuda')
    bias = torch.randn(*normalized_shape, device='cuda')
    
    my_module = build_kernel()
    output = my_module.forward(x, weight, bias)
    ref = reference_layernorm(x, weight, bias, 1e-5)
    torch.cuda.synchronize()
    
    # We expect that the fixed mask in __shfl_down_sync could lead to an incorrect reduction.
    diff = (output - ref).abs().max()
    assert diff > 1e-3, f"Kernel with small normalized_size did not trigger the expected error. Max diff: {diff}"
