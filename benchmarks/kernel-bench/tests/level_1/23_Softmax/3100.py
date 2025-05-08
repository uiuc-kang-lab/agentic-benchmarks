
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="softmax_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

def reference_softmax(x: torch.Tensor) -> torch.Tensor:
    # Compute softmax using pytorch's API for comparison.
    # We assume softmax is computed on dim 1 as expected.
    return torch.softmax(x, dim=1)

# Test case 1: Trigger mismatch in shared memory allocation by passing an unsupported block_size.
def test_block_size_mismatch():
    # Using an unsupported block size (e.g., 300) should fall back to default (256) kernel launch,
    # but the shared memory allocation gets computed with block_size=300.
    # This likely leads to numerical differences in the output.
    cuda_module = build_kernel()
    batch_size = 8
    num_features = 1024  # Multiple of block size is not needed here.
    x = torch.randn(batch_size, num_features, device="cuda", dtype=torch.float32)
    # Call the CUDA kernel with an unsupported block size
    y = cuda_module.forward(x, 300)
    # Compare with PyTorch's softmax. If the kernel is mismatched, the result may be wrong.
    y_ref = reference_softmax(x)
    # We expect a discrepancy because of the shared memory size mismatch.
    assert not torch.allclose(y, y_ref, atol=1e-4), \
           "Test failed: Kernel output unexpectedly matches the reference output with an unsupported block size."

# Test case 2: Input tensor with non-float32 type. Should raise an error in the forward C++ function.
def test_invalid_dtype():
    cuda_module = build_kernel()
    batch_size = 8
    num_features = 1024
    # Create a tensor with float64 which should trigger a TORCH_CHECK error.
    x = torch.randn(batch_size, num_features, device="cuda", dtype=torch.float64)
    with pytest.raises(RuntimeError):
        cuda_module.forward(x, 256)

# Test case 3: Input tensor with wrong dimensions. The kernel expects 2D input.
def test_invalid_dimensions():
    cuda_module = build_kernel()
    # Create a 3D tensor, which should fail the dimension check.
    x = torch.randn(4, 8, 16, device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        cuda_module.forward(x, 256)

# Test case 4: Valid input but with a number of features not divisible by the block size.
# This may work correctly in many cases, but we include it to ensure kernel generality.
def test_non_divisible_num_features():
    cuda_module = build_kernel()
    batch_size = 8
    num_features = 10007   # Not divisible by any typical block size
    x = torch.randn(batch_size, num_features, device="cuda", dtype=torch.float32)
    y = cuda_module.forward(x, 256)
    y_ref = reference_softmax(x)
    assert torch.allclose(y, y_ref, atol=1e-4), "Kernel output differs from reference softmax for non-divisible num_features."

if __name__ == "__main__":
    pytest.main([__file__])
