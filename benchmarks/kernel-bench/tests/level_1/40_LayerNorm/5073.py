
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the CUDA extension from 'kernel.cu'
    cuda_module = load(
        name="test_layernorm_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: The kernel does not support half precision (float16) inputs.
def test_half_precision_not_supported():
    my_module = build_kernel()
    # Create input tensors in half precision
    batch = 4
    norm_dim = 128
    # For simplicity, pretend the input is 2D [batch, normalized_size]
    x = torch.randn(batch, norm_dim, device="cuda", dtype=torch.float16)
    # weight and bias must have size norm_dim
    weight = torch.randn(norm_dim, device="cuda", dtype=torch.float16)
    bias   = torch.randn(norm_dim, device="cuda", dtype=torch.float16)
    
    # Expect kernel dispatching to fail because AT_DISPATCH_FLOATING_TYPES excluded float16.
    with pytest.raises(RuntimeError):
        out = my_module.forward(x, weight, bias)

# Issue 2: The kernel assumes a flattened normalized dimension.
# If weight and bias are provided in a multi–dimensional shape different from a flat vector,
# the kernel indexing will be incorrect. We simulate this by creating weight and bias with an extra dimension.
def test_non_flat_normalized_shape():
    my_module = build_kernel()
    batch = 2
    # For this test, let the normalized shape be multi–dimensional
    norm_shape = (4, 8)
    normalized_size = 4 * 8
    # Create an input tensor with extra batch dimensions preceding the normalized dimensions.
    x = torch.randn(batch, *norm_shape, device="cuda", dtype=torch.float32)
    # Instead of flattening weight and bias, we provide them in the same multi–dimensional shape.
    weight = torch.randn(*norm_shape, device="cuda", dtype=torch.float32)
    bias   = torch.randn(*norm_shape, device="cuda", dtype=torch.float32)
    
    # The kernel expects weight and bias to be flat arrays of length normalized_size.
    # Thus, using multi–dimensional weight and bias should produce an error or incorrect behavior.
    # We check that either an error is raised or the output is not close to a reference LayerNorm.
    # For a fair comparison, we compute the reference using the flattened parameters.
    weight_flat = weight.reshape(normalized_size)
    bias_flat = bias.reshape(normalized_size)
    
    # Compute reference LayerNorm using PyTorch's built-in functionality for flat weight/bias.
    x_flat = x.reshape(batch, normalized_size)
    mean = x_flat.mean(1, keepdim=True)
    var = x_flat.var(1, unbiased=False, keepdim=True)
    inv_std = 1.0 / torch.sqrt(var + 1e-5)
    ref = ((x_flat - mean) * inv_std) * weight_flat + bias_flat
    ref = ref.reshape_as(x)
    
    # When calling our kernel, we simulate the mistake by passing multi–dimensional weight/bias.
    out = my_module.forward(x, weight, bias)
    
    # The output is expected to be different from the reference because the indexing in the kernel is incorrect.
    with pytest.raises(AssertionError):
        torch.testing.assert_allclose(out, ref, atol=1e-5)

# Issue 3: The kernel uses static shared memory arrays of size 32 (one per warp).
# This test simulates a situation where a large normalized_size forces a large number of threads.
# Although current CUDA hardware restricts threads per block to 1024 (32 warps), we can mimic an environment
# where the kernel is launched with a custom block size larger than 1024 if possible.
# Here, we check if the kernel throws an error when launched with an intentionally risky thread count.
def test_excessive_threads_per_block():
    my_module = build_kernel()
    # Create an input tensor with a large normalized dimension.
    batch = 1
    # We pick a normalized size that's huge; but note the kernel's thread count is computed as min(normalized_size, 1024)
    # so to simulate the issue we have to manually override the launch configuration.
    # Since we cannot easily override the launch in the provided interface, we simulate by expecting the normal code path.
    # This test is more of a placeholder to indicate that the kernel is not ready for block sizes larger than 1024.
    norm_size = 2048  # more than 1024 elements per instance: kernel will use 1024 threads.
    x = torch.randn(batch, norm_size, device="cuda", dtype=torch.float32)
    weight = torch.randn(norm_size, device="cuda", dtype=torch.float32)
    bias   = torch.randn(norm_size, device="cuda", dtype=torch.float32)
    
    # In current conditions, the kernel will clamp blockDim.x to 1024.
    # Thus, we simply run it and check that the output dimensions match.
    out = my_module.forward(x, weight, bias)
    assert out.shape == x.shape, "Output shape must match input shape when using large normalized dimension."

