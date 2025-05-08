
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Utility function to build the kernel extension.
def build_kernel():
    # Force a rebuild to pick up changes in kernel.cu if necessary.
    # Note: adjust include paths as needed.
    cuda_module = load(
        name="rms_norm_ext",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        is_python_module=False,
        verbose=True,
    )
    return cuda_module

# Issue 1: Hard-coded shared memory size and block configuration.
# We simulate this by attempting to use a tensor size that might require a different thread block configuration.
def test_shared_memory_and_block_size_issue():
    # Create a tensor where num_features is much less than 256.
    # For example, num_features = 10. Even though the kernel launches 256 threads,
    # most threads will iterate over an empty range.
    batch_size = 4
    num_features = 10
    spatial = (16, 16)
    eps = 1e-5
    x = torch.randn(batch_size, num_features, *spatial, device="cuda", dtype=torch.float32)
    kernel = build_kernel()
    # This call uses the kernel's hard-coded block size.
    y = kernel.forward(x, eps)
    # Compute reference result using PyTorch’s built-in operations.
    rms = torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + eps)
    y_ref = x / rms
    # The result might be computed correctly, but if one later changes the block config,
    # the fixed shared memory array (size 256) may cause memory corruption.
    assert torch.allclose(y, y_ref, atol=1e-5), "Output differs, potential shared memory/block size issue."

# Issue 2: Assumption of contiguous and canonical memory layout.
def test_non_contiguous_input():
    batch_size = 4
    num_features = 32
    spatial = (8, 8)
    eps = 1e-5
    x = torch.randn(batch_size, num_features, *spatial, device="cuda", dtype=torch.float32)
    # Create a non-contiguous tensor by transposing two dimensions
    x_non_contig = x.transpose(1, 2)
    # Note: The kernel expects contiguous memory in a specific layout,
    # so passing a non-contiguous tensor may lead to incorrect results.
    kernel = build_kernel()
    try:
        y = kernel.forward(x_non_contig, eps)
    except Exception as e:
        pytest.skip("Kernel did not support non-contiguous input (as expected); ensure caller makes tensor contiguous.")
    # If no exception is raised, check for correctness.
    # Force contiguous version and compare.
    y_contig = kernel.forward(x_non_contig.contiguous(), eps)
    rms = torch.sqrt(torch.mean(x_non_contig.contiguous()**2, dim=1, keepdim=True) + eps)
    y_ref = x_non_contig.contiguous() / rms
    assert not torch.allclose(y, y_ref, atol=1e-5), "Kernel incorrectly handled non-contiguous input."

# Issue 3: Limited type dispatch.
def test_unsupported_dtype():
    batch_size = 4
    num_features = 32
    spatial = (8, 8)
    eps = 1e-5
    # Use bfloat16 which is not handled by AT_DISPATCH_FLOATING_TYPES_AND_HALF.
    x = torch.randn(batch_size, num_features, *spatial, device="cuda", dtype=torch.bfloat16)
    kernel = build_kernel()
    with pytest.raises(RuntimeError):
        _ = kernel.forward(x, eps)

# Issue 4: Use of __fdiv_rd leading to potential numerical differences.
def test_nonstandard_division_rounding():
    batch_size = 8
    num_features = 64
    spatial = (16, 16)
    eps = 1e-5
    x = torch.randn(batch_size, num_features, *spatial, device="cuda", dtype=torch.float32)
    kernel = build_kernel()
    # Run the kernel.
    y = kernel.forward(x, eps)
    # Compute the expected result with standard division.
    rms = torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + eps)
    y_ref = x / rms
    # The use of __fdiv_rd can cause slight discrepancies;
    # we expect a difference larger than typical roundoff if the divergence happens.
    max_diff = (y - y_ref).abs().max().item()
    assert max_diff > 1e-7, "Numerical differences expected due to __fdiv_rd but differences are too small."

# Issue 5: Lack of error checking after kernel launch.
def test_kernel_launch_error_check():
    # We simulate an error by providing a tensor with zero spatial elements.
    batch_size = 4
    num_features = 32
    spatial = (0,)  # zero elements per batch region
    eps = 1e-5
    x = torch.randn(batch_size, num_features, *spatial, device="cuda", dtype=torch.float32)
    kernel = build_kernel()
    # The kernel may launch without explicit error checking.
    # Here we check that either the kernel fails, or it produces an output tensor with the expected shape.
    y = kernel.forward(x, eps)
    if x.numel() == 0:
        # Either way, it is better to get a runtime error rather than silently produce garbage.
        assert y.numel() == 0, "Kernel did not handle zero-element input correctly."
    else:
        pytest.skip("Could not simulate kernel launch error with zero-element input.")

