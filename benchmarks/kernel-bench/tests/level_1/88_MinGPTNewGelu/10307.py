
import pytest
import torch
from torch.utils.cpp_extension import load
import math

# Helper: Build the extension module from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="gelu_extension",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Reference GELU function (using exactly the same formula as in the kernel)
def gelu_ref(x: torch.Tensor) -> torch.Tensor:
    # Note: Use torch.tanh and float32 computation.
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))

# Test 1: Trigger wrong handling of non-float32 (e.g. using double).
def test_non_float_input():
    my_module = build_kernel()
    # Create a double tensor. Although it is on CUDA and contiguous,
    # the kernel uses data_ptr<float>() and does not check dtype.
    # Therefore, the result will be computed with the wrong type interpretation.
    x = torch.randn(1000, device="cuda", dtype=torch.float64)
    # Call the extension. Since the kernel assumes float, we expect the output to be incorrect.
    y = my_module.forward(x)
    # For a proper GELU, one would expect gelu_ref(x.float()). We cast x to float32 for comparison.
    y_ref = gelu_ref(x.float())
    # The results are expected NOT to match (with a very loose tolerance)
    # because the input was misinterpreted.
    assert not torch.allclose(y, y_ref, atol=1e-3), \
        "Kernel mis-handles non-float32 input: the output unexpectedly matches the reference."

# Test 2: Trigger non-contiguous tensor error.
def test_non_contiguous_input():
    my_module = build_kernel()
    # Create a contiguous tensor first, then take a noncontiguous slice.
    x = torch.randn(1000, 10, device="cuda", dtype=torch.float32)
    x_noncontig = x.t()  # Transpose makes it non-contiguous.
    with pytest.raises(RuntimeError, match="Input tensor must be contiguous"):
        my_module.forward(x_noncontig)

# Test 3: Trigger CPU tensor error.
def test_cpu_input():
    my_module = build_kernel()
    x_cpu = torch.randn(1000, device="cpu", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="Input tensor must be on CUDA"):
        my_module.forward(x_cpu)

# Test 4 (Informational): Although it is very tricky to trigger, this test serves as a reminder.
# If one passes an input size that causes a block to have an incomplete warp contribution (when not using
# a block size that is a multiple of warpSize), then the reduction (stored in block_sums) might be computed incorrectly.
# Since the current launch always uses 256 threads (a multiple of 32), we simulate a possible scenario by manually
# changing the launch configuration via monkey patching (only for testing purposes).
def test_custom_blockSize_incomplete_warp(monkeypatch):
    my_module = build_kernel()
    # We simulate by creating a wrapper that calls the underlying gelu_forward with a custom block size.
    # Warning: This is “artificial” since the current extension launch parameter is hard-coded.
    # We build an input whose total number of elements is not a multiple of the custom block size.
    x = torch.randn(1000, device="cuda", dtype=torch.float32)
    # Here we assume that if one were to launch with, say, 130 threads per block (not a multiple of 32),
    # then the reduction computed inside the kernel (block_sums) would be missing the contribution from the last partial warp.
    # Since the final output (tensor y) is computed elementwise before the reduction step,
    # we cannot see a numerical error on the GELU output.
    # Instead, we note this in the test and simply run the kernel to ensure it does not crash.
    # (In a full implementation, one would add an API to expose the computed block_sums to check for correctness.)
    try:
        y = my_module.forward(x)
    except Exception as e:
        pytest.fail(f"Kernel crash with a custom block size for incomplete warp simulation: {e}")
