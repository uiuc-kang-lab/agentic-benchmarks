
import pytest
import torch
from torch.utils.cpp_extension import load

# Utility to compile and load the cuda extension from kernel.cu
def build_kernel():
    return load(
        name="optimized_loss",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )

# Test case 1: Pass a tensor of type float64 to trigger the float32 assumption.
def test_non_float32_dtype():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    mod = build_kernel()
    # Create tensors in double precision
    predictions = torch.randn(4096, device="cuda", dtype=torch.float64)
    targets = torch.randn(4096, device="cuda", dtype=torch.float64)
    with pytest.raises(RuntimeError) as excinfo:
        # Expect that the kernel fails (likely with an invalid argument or type error)
        mod.forward(predictions, targets)
    assert "expected" in str(excinfo.value).lower() or "float" in str(excinfo.value).lower()

# Test case 2: Pass non-contiguous tensors to trigger the contiguous requirement.
def test_non_contiguous_tensors():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    mod = build_kernel()
    # Create contiguous tensors and then get a non-contiguous view by transposing a 2D tensor.
    base = torch.randn(32, 128, device="cuda", dtype=torch.float32)
    predictions = base.t()  # transpose makes it non-contiguous
    targets = base.t()
    with pytest.raises(RuntimeError) as excinfo:
        mod.forward(predictions, targets)
    assert "contiguous" in str(excinfo.value).lower()

# Test case 3: Pass CPU tensors to trigger the CUDA device requirement.
def test_cpu_inputs():
    mod = build_kernel()
    predictions = torch.randn(4096, device="cpu", dtype=torch.float32)
    targets = torch.randn(4096, device="cpu", dtype=torch.float32)
    with pytest.raises(RuntimeError) as excinfo:
        mod.forward(predictions, targets)
    assert "cuda" in str(excinfo.value).lower()
