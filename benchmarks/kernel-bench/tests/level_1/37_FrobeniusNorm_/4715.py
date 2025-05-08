
import pytest
import torch
from torch.utils.cpp_extension import load

# Utility function to build the extension from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Using 'min' improperly might cause a compile-time or runtime problem.
# (This test simply builds the kernel. If compilation fails due to min, this test will error.)
def test_min_namespace_issue():
    try:
        mod = build_kernel()
    except Exception as e:
        pytest.fail(f"Compilation failed due to min function issue: {e}")

# Issue 2: Lack of synchronization between kernel launch and cudaMemcpy.
# Provide a relatively large input to encourage use of many blocks.
def test_synchronization_issue():
    mod = build_kernel()
    # Create a large tensor so that multiple kernel launches occur.
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)
    # Call the forward function.
    # If the lack of proper synchronization causes race conditions,
    # the computed norm may be incorrect and the output won’t be consistent.
    out = mod.forward(x)
    # Compute expected normalized tensor on CPU.
    norm = torch.norm(x, p='fro')
    # When norm is nonzero, the result of division should be close to the normalized tensor.
    # Allow for some tolerance which might be affected by race conditions.
    expected = x / norm
    assert torch.allclose(out, expected, atol=1e-5), "Output does not match expected normalized tensor (possible sync issue)"

# Issue 3: Hard-coded blockDim assumption. We trigger this by trying to run the kernel on a nonstandard tensor shape.
# Although the kernel always launches with threads=256, different input sizes (especially very small ones)
# might expose issues in edge-case indexing in the reduction.
def test_blockDim_assumption():
    mod = build_kernel()
    # Create a small tensor where number of elements is less than the block size.
    x = torch.randn(10, device="cuda", dtype=torch.float32)
    out = mod.forward(x)
    norm = torch.norm(x, p='fro')
    expected = x / norm
    assert torch.allclose(out, expected, atol=1e-5), "Output incorrect for small tensor (blockDim assumption issue?)"

# Issue 4: No check for division by zero.
# Provide an input tensor that is entirely zeros.
def test_divide_by_zero():
    mod = build_kernel()
    x = torch.zeros(100, device="cuda", dtype=torch.float32)
    out = mod.forward(x)
    # When norm is zero, division would produce NaNs or Infs.
    # We check that this behavior occurs.
    if torch.isnan(out).any() or torch.isinf(out).any():
        pass  # expected behavior, division by zero happened
    else:
        pytest.fail("Division by zero did not yield NaN or Inf as expected.")

# Issue 5: No error checking after CUDA API calls.
# We simulate a bad input that violates the kernel's assumptions: 
# non-contiguous tensor and wrong data type.
def test_input_validations():
    mod = build_kernel()
    # Non-contiguous tensor
    x = torch.randn(64, 64, device="cuda", dtype=torch.float32).t().clone()  # .t() makes it non-contiguous
    with pytest.raises(RuntimeError) as excinfo:
        mod.forward(x)
    assert "contiguous" in str(excinfo.value).lower(), "Kernel did not check for contiguous tensor"

    # Wrong scalar type: double instead of float32
    x = torch.randn(64, 64, device="cuda", dtype=torch.float64)
    with pytest.raises(RuntimeError) as excinfo:
        mod.forward(x)
    assert "float32" in str(excinfo.value).lower(), "Kernel did not check for float32 type"

if __name__ == "__main__":
    pytest.main([__file__])
