
import torch
import pytest
from torch.utils.cpp_extension import load

# Build our custom CUDA extension that includes kernel.cu
def build_kernel():
    cuda_module = load(
        name="optimized_hardsigmoid",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Reference hard sigmoid computation (identical to PyTorch's functional version)
def ref_hardsigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp((x + 3) / 6, min=0, max=1)

# Issue 1: Test misaligned memory accesses.
# We force a misaligned pointer by allocating an extra element and taking a slice starting from index 1.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_misaligned_input():
    cuda_mod = build_kernel()
    # Allocate extra element so we can get an offset slice.
    base = torch.randn(1025, device="cuda", dtype=torch.float32)
    # Get a slice, which will be contiguous but its data_ptr is offset by 4 bytes.
    x = base.narrow(0, 1, 1024)
    # Expect the result from our kernel to be close to the reference hard sigmoid.
    y_kernel = cuda_mod.forward(x)
    torch.cuda.synchronize()
    y_ref = ref_hardsigmoid(x)
    # Using a more lenient tolerance because misaligned accesses may yield wrong results.
    assert not torch.allclose(y_kernel, y_ref, atol=1e-5), (
        "Test for misaligned memory access did not trigger the expected issue: output matched the reference."
    )

# Issue 2: Test non-contiguous tensor.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_non_contiguous_input():
    cuda_mod = build_kernel()
    # Create a 2D tensor and then take its transpose so it becomes non-contiguous.
    x = torch.randn(64, 128, device="cuda", dtype=torch.float32)
    x_t = x.transpose(0, 1)  # non-contiguous tensor
    # Even though the tensor is non-contiguous, our kernel will use the same data_ptr view.
    y_kernel = cuda_mod.forward(x_t)
    torch.cuda.synchronize()
    y_ref = ref_hardsigmoid(x_t)
    # The computed result may be erroneous due to non-contiguity.
    assert not torch.allclose(y_kernel, y_ref, atol=1e-5), (
        "Test for non-contiguous input did not trigger the expected issue: output matched the reference."
    )

# Issue 3: Test unsupported half precision.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_half_precision_input():
    cuda_mod = build_kernel()
    # Create a tensor in half precision.
    x = torch.randn(1024, device="cuda", dtype=torch.float16)
    # The dispatch in the kernel does not support half, so we expect an exception.
    with pytest.raises(RuntimeError):
        _ = cuda_mod.forward(x)

# Issue 4: Test fallback path for non-multiple-of-4 lengths.
# When numel % 4 != 0 the kernel falls back to the scalar version.
# This may be functionally correct but represents a potential performance issue.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_non_multiple_of_four():
    cuda_mod = build_kernel()
    # Create a tensor with length not a multiple of 4, e.g. 1025 elements.
    x = torch.randn(1025, device="cuda", dtype=torch.float32)
    y_kernel = cuda_mod.forward(x)
    torch.cuda.synchronize()
    y_ref = ref_hardsigmoid(x)
    # Although numerically the fallback should compute the correct result,
    # if the fallback were mistakenly implemented, the error would be visible.
    assert torch.allclose(y_kernel, y_ref, atol=1e-5), (
        "Fallback kernel for non-multiple-of-4 elements produced an incorrect result."
    )
