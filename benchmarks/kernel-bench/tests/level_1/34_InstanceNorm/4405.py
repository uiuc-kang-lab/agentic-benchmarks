
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="instance_norm_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return cuda_module

# Issue 1: Noncontiguous input may lead to mis‐aligned memory accesses.
def test_non_contiguous_input():
    kernel = build_kernel()
    # Create a contiguous input and then force noncontiguity via transpose/slice.
    x = torch.randn(16, 64, 256, 256, device="cuda", dtype=torch.float32)
    x_noncontig = x.transpose(1, 2)  # make it noncontiguous
    weight = torch.ones(64, device="cuda", dtype=torch.float32)
    bias = torch.zeros(64, device="cuda", dtype=torch.float32)
    eps = 1e-5
    # The kernel expects a contiguous (and properly aligned) tensor.
    with pytest.raises(RuntimeError):
        # Likely the reinterpret_cast will produce an error (or undefined behavior).
        y = kernel.forward(x_noncontig, weight, bias, eps)
        torch.cuda.synchronize()

# Issue 2: Input tensor type is not verified. Passing a non–float32 (double) tensor should fail.
def test_non_float32_input():
    kernel = build_kernel()
    x = torch.randn(16, 64, 256, 256, device="cuda", dtype=torch.float64)  # double type
    weight = torch.ones(64, device="cuda", dtype=torch.float32)  # still float32
    bias = torch.zeros(64, device="cuda", dtype=torch.float32)
    eps = 1e-5
    with pytest.raises(RuntimeError):
        # This should trigger either a failed cast or incorrect memory access.
        y = kernel.forward(x, weight, bias, eps)
        torch.cuda.synchronize()

# Issue 3: The kernel requires that height*width is divisible by 4.
def test_invalid_hw_input():
    kernel = build_kernel()
    # Use spatial dimensions that do not yield a total number of elements divisible by 4.
    # For instance, height=255, width=255 -> 65025 elements per channel, which is not divisible by 4.
    x = torch.randn(16, 64, 255, 255, device="cuda", dtype=torch.float32)
    weight = torch.ones(64, device="cuda", dtype=torch.float32)
    bias = torch.zeros(64, device="cuda", dtype=torch.float32)
    eps = 1e-5
    with pytest.raises(RuntimeError) as excinfo:
        y = kernel.forward(x, weight, bias, eps)
        torch.cuda.synchronize()
    assert "HW must be multiple of 4" in str(excinfo.value)

# Issue 4: The use of "#pragma unroll 4" with a runtime value (HW4) may yield incorrect results 
# when HW4 is not a multiple of 4. This test creates an input where (H*W)/4 is not a multiple of 4.
def test_loop_unroll_issue():
    kernel = build_kernel()
    # Use spatial dimensions such that H*W is divisible by 4 but (H*W)/4 is not.
    # For example, height=5 and width=4 give 20 elements per channel, so HW4=5.
    x = torch.randn(2, 3, 5, 4, device="cuda", dtype=torch.float32)
    weight = torch.randn(3, device="cuda", dtype=torch.float32)
    bias = torch.randn(3, device="cuda", dtype=torch.float32)
    eps = 1e-5
    
    # Run the custom kernel
    y_custom = kernel.forward(x, weight, bias, eps)
    
    # Use PyTorch's InstanceNorm2d for reference
    # (Note: PyTorch's InstanceNorm2d applies affine transformation based on weight and bias,
    # so we must initialize it with identical parameters.)
    instance_norm = torch.nn.InstanceNorm2d(num_features=3, eps=eps, affine=True)
    with torch.no_grad():
        instance_norm.weight.copy_(weight)
        instance_norm.bias.copy_(bias)
    y_py = instance_norm(x)
    
    # The outputs are expected to be (almost) equal.
    # If the unrolling issue is present the outputs may differ noticeably.
    assert torch.allclose(y_custom, y_py, atol=1e-4), \
        f"Custom kernel output differs from PyTorch InstanceNorm2d output! Max diff: {(y_custom-y_py).abs().max()}"

if __name__ == "__main__":
    pytest.main([__file__])
