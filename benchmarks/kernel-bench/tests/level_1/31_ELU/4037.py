
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Trigger issue with non-float32 input.
def test_non_float32_input():
    # Create a double tensor (float64) which the kernel is not designed for.
    x = torch.randn(16, 16384, dtype=torch.float64, device="cuda")
    module = build_kernel()
    # Since the kernel always casts data_ptr<float>(), the output will be nonsensical.
    # We check that the output does not match the expected ELU computed on float64.
    out = module.forward(x)
    # Compute expected result with PyTorch, converting x to float for a fair comparison.
    expected = torch.nn.functional.elu(x.float(), alpha=1.0)
    # The absolute difference between out and expected when cast to float will be huge.
    with pytest.raises(AssertionError):
        assert torch.allclose(out, expected, atol=1e-5), "Kernel accepted non-float32 dtype unexpectedly!"

# Test 2: Trigger issue with non-contiguous input.
def test_non_contiguous_input():
    # Create a contiguous tensor first.
    x = torch.randn(16, 16384, device="cuda", dtype=torch.float32)
    # Make it non-contiguous by transposing (if 2D) or slicing.
    # Since the tensor shape is not square generally, we can force non-contiguity via slicing.
    y = x[:, ::2]
    assert not y.is_contiguous(), "Test tensor is unexpectedly contiguous."
    module = build_kernel()
    with pytest.raises(torch._C._RuntimeError) as excinfo:
        _ = module.forward(y)
    # Check that the error message complains about contiguity.
    assert "must be contiguous" in str(excinfo.value)

# Test 3: Compilation check for proper use of min
def test_compilation_min_usage():
    """
    This test ensures that the module builds and the symbol 'forward' is available.
    The use of 'min' in kernel.cu without proper qualification would cause a compile-time error.
    """
    try:
        module = build_kernel()
        # If the module loads, it means the min usage issue is not breaking compilation.
        # We run a simple test.
        x = torch.randn(16, 16384, device="cuda", dtype=torch.float32)
        out = module.forward(x)
        expected = torch.nn.functional.elu(x, alpha=1.0)
        assert torch.allclose(out, expected, atol=1e-5), "Kernel forward computation is incorrect."
    except Exception as e:
        pytest.fail("Kernel compilation failed (possibly due to improper use of min): " + str(e))
