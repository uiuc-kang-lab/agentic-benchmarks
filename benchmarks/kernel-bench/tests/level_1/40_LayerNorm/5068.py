
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="custom_layernorm",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Helper function to compute reference LayerNorm using PyTorch
def reference_layernorm(x, weight, bias, eps=1e-5):
    # Assume normalization is over the last dimension.
    normalized_shape = weight.shape
    return torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)

# Issue 1: Non power-of-two normalized_size.
def test_non_power_of_two_reduction():
    # Use normalized_shape of size 50 (not a power of two)
    batch_size = 16
    normalized_size = 50
    # Create an input tensor where each row is normalized.
    x = torch.randn(batch_size, normalized_size, device="cuda", dtype=torch.float32)
    weight = torch.randn(normalized_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(normalized_size, device="cuda", dtype=torch.float32)
    custom_module = build_kernel()
    custom_out = custom_module.forward(x, weight, bias, 1e-5)
    ref_out = reference_layernorm(x, weight, bias, 1e-5)
    # We expect the results to differ because of the flawed reduction.
    assert not torch.allclose(custom_out, ref_out, atol=1e-4), (
        "Test failed to trigger the non-power-of-two reduction issue: "
        "custom kernel output matches reference output unexpectedly."
    )

# Issue 2: Non-contiguous input tensor.
def test_non_contiguous_input():
    # Create a tensor and then transpose it to be non-contiguous in the normalized dimension.
    batch_size = 16
    normalized_size = 64
    # Create a 2D tensor and then transpose so that the normalized dimension is not contiguous.
    x = torch.randn(normalized_size, batch_size, device="cuda", dtype=torch.float32).t().contiguous(False)
    weight = torch.randn(normalized_size, device="cuda", dtype=torch.float32)
    bias = torch.randn(normalized_size, device="cuda", dtype=torch.float32)
    custom_module = build_kernel()
    try:
        custom_out = custom_module.forward(x, weight, bias, 1e-5)
        # If the kernel runs, check that the result is different from PyTorch's layer_norm.
        ref_out = reference_layernorm(x.contiguous(), weight, bias, 1e-5)
        assert not torch.allclose(custom_out, ref_out, atol=1e-4), (
            "Test failed to trigger the non-contiguous input issue: "
            "custom kernel output unexpectedly matches reference output on a non-contiguous input."
        )
    except RuntimeError as e:
        # If a runtime error arises due to non-contiguity, that also indicates an issue.
        pytest.skip("Kernel does not handle non-contiguous inputs (skipping test), which confirms the issue.")

# Issue 3: Lack of error checking for kernel launch; testing with mismatched shapes.
def test_shape_mismatch():
    # Prepare input tensor where normalized dimension size does not match the size of weight and bias.
    batch_size = 16
    normalized_size = 64
    x = torch.randn(batch_size, normalized_size, device="cuda", dtype=torch.float32)
    # Create weight and bias with an incorrect size (e.g., 32 instead of 64)
    weight = torch.randn(32, device="cuda", dtype=torch.float32)
    bias = torch.randn(32, device="cuda", dtype=torch.float32)
    custom_module = build_kernel()
    with pytest.raises(RuntimeError):
        # This should raise a runtime error due to shape mismatch.
        custom_module.forward(x, weight, bias, 1e-5)
