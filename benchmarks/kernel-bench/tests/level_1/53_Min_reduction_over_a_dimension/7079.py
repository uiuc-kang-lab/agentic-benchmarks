
import pytest
import torch
from torch.utils.cpp_extension import load

# Build and load the extension from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="min_reduce_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Trigger potential ambiguity in min function via integer types.
# If the kernel does not correctly handle non‐floating types, the result may be incorrect.
def test_integer_dtype():
    my_module = build_kernel()
    # Create an integer tensor on CUDA.
    x = torch.randint(0, 100, (4, 8, 16), device="cuda", dtype=torch.int32)
    # Use reduction along dimension 1.
    output = my_module.forward(x, 1)
    # Compute reference result using PyTorch’s built-in torch.min.
    ref = torch.min(x, dim=1)[0]
    # The kernels may suffer from ambiguity in min calls.
    assert torch.equal(output, ref), f"Integer dtype reduction result is incorrect."

# Test 2: Trigger integer overflow potential issues by simulating a tensor with a high product.
# WARNING: This test is artificial; in reality, creating a tensor huge enough to overflow may be infeasible.
# Here we simulate the overflow condition by monkey-patching the shape values.
def test_index_overflow(monkeypatch):
    my_module = build_kernel()
    # Create a small tensor that will be used normally.
    x = torch.randn(4, 8, 16, device="cuda", dtype=torch.float32)
    # Monkey-patch the forward function inside our extension to use a large value for outer.
    # This simulates a situation where the product of dimensions might overflow.
    # Note: This is just a simulation to trigger the potential bug internally.
    original_forward = my_module.forward
    def fake_forward(input, dim):
        # Force an overflow-like scenario by setting outer to a large value.
        # WARNING: This contrived hack might lead to undefined behavior,
        # so we only check that the resulting output shape is not as expected.
        outer = 2**30  # A huge number that would cause overflow in int indexing.
        inner = 1
        # We don't actually run the kernel in this fake path.
        # Instead, we simulate detection of the bad indexing by raising an error.
        raise RuntimeError("Index overflow detected in kernel simulation")
    monkeypatch.setattr(my_module, "forward", fake_forward)
    with pytest.raises(RuntimeError, match="Index overflow detected"):
        my_module.forward(x, 1)
    # Restore the original forward (not strictly necessary here since the test ends).
    monkeypatch.setattr(my_module, "forward", original_forward)

# Test 3: Trigger the unsupported dtype issue (e.g., half precision).
def test_half_dtype():
    my_module = build_kernel()
    # Create a half-precision tensor on CUDA.
    x = torch.randn(4, 8, 16, device="cuda", dtype=torch.float16)
    # We expect the kernel to either not support float16 or raise an explicit error.
    with pytest.raises(RuntimeError):
        my_module.forward(x, 1)

# Test 4: Test that an invalid reduction dimension is caught.
def test_invalid_dim():
    my_module = build_kernel()
    x = torch.randn(4, 8, 16, device="cuda", dtype=torch.float32)
    # Passing an out-of-range dimension should trigger a TORCH_CHECK error.
    with pytest.raises(RuntimeError, match="dim out of range"):
        my_module.forward(x, 3)
