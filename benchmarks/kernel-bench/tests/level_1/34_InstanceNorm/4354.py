
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build the CUDA extension module from kernel.cu
def build_kernel():
    module = load(
        name="instance_norm_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return module

# Test 1: Pass a tensor of type double (float64) to trigger type incompatibility.
def test_input_tensor_type():
    kernel_module = build_kernel()
    
    # Create input tensor of type double
    N, C, H, W = 4, 8, 32, 32
    x = torch.randn(N, C, H, W, dtype=torch.float64, device='cuda')
    weight = torch.ones(C, dtype=torch.float64, device='cuda')
    bias = torch.zeros(C, dtype=torch.float64, device='cuda')
    eps = 1e-5

    with pytest.raises(RuntimeError):
        # This should fail because the kernel expects float32.
        kernel_module.forward(x, weight, bias, eps)

# Test 2: Provide only weight but not bias to trigger the issue with affine parameters handling.
def test_weight_only_provided():
    kernel_module = build_kernel()

    # Create input tensor in float32.
    N, C, H, W = 4, 8, 32, 32
    x = torch.randn(N, C, H, W, dtype=torch.float32, device='cuda')
    # Provide weight but simulate missing bias by an empty tensor.
    weight = torch.ones(C, dtype=torch.float32, device='cuda')
    bias = torch.tensor([], dtype=torch.float32, device='cuda')  # Empty bias tensor

    eps = 1e-5

    # The forward function checks if bias.defined() and bias.numel()>0.
    # So here no scaling will be applied even though weight is provided.
    # We test that the output does not match the expected normalized result when weight were applied.
    y = kernel_module.forward(x, weight, bias, eps)
    
    # Compute reference normalization without scaling (affine transformation)
    # because our kernel will skip scaling if bias is absent.
    N_val, C_val, H_val, W_val = x.shape
    y_ref = torch.empty_like(x)
    for n in range(N_val):
        for c in range(C_val):
            instance = x[n, c]
            mean = instance.mean()
            var = instance.var(unbiased=False)
            y_ref[n, c] = (instance - mean) / torch.sqrt(var + eps)
    # Since weight is ignored when bias is not provided, y should equal y_ref.
    assert torch.allclose(y, y_ref, atol=1e-4), "Kernel did not perform expected normalization when only weight is provided."

# Test 3: Create a situation where the block reduction may exceed 32 warps.
# We simulate this by launching a kernel with a larger block.
# To do so, we create a fake input with a very high spatial dimension such that we desire a block configuration larger than 16x16.
# Although the kernel uses fixed block dimensions in forward(), this test mimics the case where a developer might try to use larger blocks.
def test_large_spatial_instance():
    kernel_module = build_kernel()

    # Construct a tensor with extremely large spatial dimensions.
    N, C, H, W = 2, 4, 1024, 1024  # H*W is huge.
    x = torch.randn(N, C, H, W, dtype=torch.float32, device='cuda')
    weight = torch.ones(C, dtype=torch.float32, device='cuda')
    bias = torch.zeros(C, dtype=torch.float32, device='cuda')
    eps = 1e-5

    # Even though the forward function launches with fixed block dimensions (16x16),
    # this test serves as a warning: if a developer adjusts the block dimensions for performance,
    # the hardcoded shared memory size in the reduction may be insufficient.
    # Here we run the kernel to see if it crashes or gives wrong answers.
    y = kernel_module.forward(x, weight, bias, eps)

    # Compute reference normalization over a couple of channels and instances (for testing correctness)
    y_ref = x.clone()
    for n in range(N):
        for c in range(C):
            instance = x[n, c]
            mean = instance.mean()
            var = instance.var(unbiased=False)
            # Apply affine transform
            y_ref[n, c] = ((instance - mean) / torch.sqrt(var + eps)) * weight[c] + bias[c]
    # Due to large dimensions numerical differences may be higher; use a loose tolerance.
    assert torch.allclose(y, y_ref, atol=1e-3), "Kernel output for large spatial dimensions is incorrect."

if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])
