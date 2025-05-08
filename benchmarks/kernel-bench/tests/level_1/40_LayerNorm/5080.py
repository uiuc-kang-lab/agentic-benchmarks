
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# Build the CUDA kernel extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="layernorm_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# A helper function to run the kernel forward and compare it with PyTorch's native LayerNorm.
def reference_layernorm(x, weight, bias, eps):
    # F.layer_norm expects normalized_shape to be the last dimensions of x.
    normalized_shape = weight.shape
    return F.layer_norm(x, normalized_shape, weight, bias, eps)

# Issue 1 test: noncontiguous input and weight tensors.
# The kernel's pointer arithmetic is based on a contiguous layout.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_noncontiguous_input():
    # Create a contiguous input and then make it noncontiguous by transposing.
    batch, C, H, W = 4, 8, 16, 16
    # Use a LayerNorm normalized shape of (C, H, W)
    normalized_shape = (C, H, W)
    x = torch.randn(batch, C, H, W, device="cuda")
    # Make x noncontiguous by transposing two dimensions (and not calling contiguous())
    x_noncontig = x.transpose(1, 2)  # Now shape is [batch, H, C, W] and noncontiguous.
    
    # Weight and bias are still created as contiguous tensors matching normalized_shape.
    weight = torch.randn(normalized_shape, device="cuda")
    bias = torch.randn(normalized_shape, device="cuda")
    eps = 1e-5

    # Build the kernel module.
    module = build_kernel()

    # Run the kernel: note that the kernel expects the innermost dimension size to equal weight.numel()
    # Here, since x is noncontiguous and the kernel’s pointer arithmetic assumes contiguous storage,
    # the results will likely be wrong.
    try:
        out_kernel = module.forward(x_noncontig, weight, bias, eps)
    except RuntimeError as e:
        pytest.skip("Kernel raised an error on noncontiguous input: " + str(e))
    
    # Compute the reference result on contiguous input.
    # To simulate what the user intended the kernel to do, we use the contiguous version.
    out_ref = reference_layernorm(x_noncontig.contiguous(), weight, bias, eps)
    
    # The difference should be significant since the kernel computed on a misinterpreted layout.
    max_diff = (out_kernel - out_ref).abs().max().item()
    assert max_diff > 1e-2, (
        f"Kernel did not error on noncontiguous input, but results match too closely (max diff {max_diff})."
    )

# Issue 2 test: Handling of zero normalized_size (empty normalized dimension).
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_zero_normalized_size():
    # Create an input where the normalized dimension is empty.
    # For example: normalized_shape = (0,)
    batch = 4
    normalized_shape = (0,)
    x = torch.randn(batch, *normalized_shape, device="cuda")
    weight = torch.randn(normalized_shape, device="cuda")
    bias = torch.randn(normalized_shape, device="cuda")
    eps = 1e-5

    module = build_kernel()

    # The expected behavior is that the kernel should either gracefully handle the zero-size reduction
    # or raise an error. If it tries to divide by zero, it may produce NaNs.
    out_kernel = module.forward(x, weight, bias, eps)
    if out_kernel.numel() > 0:
        # Check for NaNs in the output.
        assert torch.isnan(out_kernel).any(), "Kernel should produce NaNs for zero normalized_size."
    else:
        pytest.skip("Output has zero elements; nothing to compare.")

# Issue 3 test: Kernel assumes one block per outer instance.
# We test with a large normalized_size that forces the kernel to use its loop over threads.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_large_normalized_size():
    # In this test, we simulate a scenario with a very large normalized dimension.
    # The kernel uses a fixed maximum of 1024 threads per block.
    batch = 2
    # Let normalized shape be large, e.g., (1024*4,) to force many iterations in the loop
    normalized_shape = (1024 * 4,)
    x = torch.randn(batch, *normalized_shape, device="cuda")
    weight = torch.randn(normalized_shape, device="cuda")
    bias = torch.randn(normalized_shape, device="cuda")
    eps = 1e-5

    module = build_kernel()

    out_kernel = module.forward(x, weight, bias, eps)
    out_ref = reference_layernorm(x, weight, bias, eps)

    # Since the kernel’s algorithm may not generalize perfectly for very large normalized_size,
    # we expect a non-negligible difference.
    max_diff = (out_kernel - out_ref).abs().max().item()
    assert max_diff > 1e-4, (
        f"Kernel output differences for large normalized size appear too small (max diff {max_diff})."
    )

# Issue 4 test: The kernel uses __ldg assuming read-only cache friendly accesses.
# We attempt to trigger a potential performance or correctness pitfall by passing tensors that are
# not optimally aligned. (This test is more qualitative: we check that the kernel result diverges.)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_unaligned_access():
    # Create a tensor with an extra slice in front to misalign the pointer addresses.
    # By slicing, we can force the internal pointer not to be aligned to what the kernel expects.
    batch = 4
    normalized_shape = (16, 16)  # small normalized size for simplicity
    x_full = torch.randn(batch + 1, *normalized_shape, device="cuda")
    # Use a slice that is likely unaligned.
    x = x_full[1:]
    weight = torch.randn(normalized_shape, device="cuda")
    bias = torch.randn(normalized_shape, device="cuda")
    eps = 1e-5

    module = build_kernel()

    out_kernel = module.forward(x, weight, bias, eps)
    out_ref = reference_layernorm(x.contiguous(), weight, bias, eps)

    # When the kernel loads data with __ldg from an unaligned address,
    # the result may be affected. We check that the outputs do not match.
    max_diff = (out_kernel - out_ref).abs().max().item()
    assert max_diff > 1e-3, (
        f"Kernel output differences for unaligned access appear too small (max diff {max_diff})."
    )
