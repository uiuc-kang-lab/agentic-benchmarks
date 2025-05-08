
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Load the CUDA extension from kernel.cu.
def build_kernel():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    cuda_file = os.path.join(this_dir, "kernel.cu")
    cuda_module = load(
        name="prod_reduce_cuda",
        sources=[cuda_file],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Trigger incorrect thread-to-output mapping and reduction logic.
# This test uses an input shape such that multiple output elements are processed in one kernel launch.
def test_incorrect_mapping():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    kernel = build_kernel()
    # Create an input tensor with shape [16, 256, 256].
    # Reduction dimension = 1, so output should have shape [16, 256]
    x = torch.randn(16, 256, 256, device="cuda", dtype=torch.float32)
    # Use the custom kernel forward
    out_kernel = kernel.forward(x, 1)
    # Use native torch.prod reduction
    out_ref = torch.prod(x, dim=1)
    # Since the kernel reduction logic is flawed, the answers should differ.
    # We expect them to differ by more than a small tolerance.
    difference = (out_kernel - out_ref).abs().max()
    assert difference > 1e-3, f"Mapping issue not triggered. Difference: {difference}"

# Test 2: Trigger incorrect boundary check in shared-memory reduction.
# This test uses an input size whose number of output elements is not a multiple of the thread block size (256).
def test_boundary_check():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    kernel = build_kernel()
    # Choose an input such that output.numel() (after reduction) is not a multiple of 256.
    # For example, input shape [5, 13] and reduction dim=1 -> output shape [5]
    x = torch.randn(5, 13, device="cuda", dtype=torch.float32)
    out_kernel = kernel.forward(x, 1)
    out_ref = torch.prod(x, dim=1)
    # Due to the flawed boundary check, results should be off.
    difference = (out_kernel - out_ref).abs().max()
    assert difference > 1e-3, f"Boundary issue not triggered. Difference: {difference}"

# Test 3: Trigger lack of support for non-float32 data types.
# This test uses a double tensor (float64). The kernel is hard-coded for float and will misinterpret data.
def test_non_float_dtype():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    kernel = build_kernel()
    # Create a double tensor
    x = torch.randn(10, 20, device="cuda", dtype=torch.float64)
    # The kernel does not check for the correct type so it will call data_ptr<float>() on a double tensor.
    # The resulting computation is expected to be completely wrong.
    out_kernel = kernel.forward(x, 1)
    out_ref = torch.prod(x, dim=1)
    difference = (out_kernel.double() - out_ref).abs().max()
    assert difference > 1e-3, f"Non float32 dtype issue not triggered. Difference: {difference}"

# Test 4: Trigger handling of non-contiguous tensors.
# This test deliberately creates a non-contiguous input tensor; the CHECK_CONTIGUOUS macro should raise an error.
def test_non_contiguous_input():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    kernel = build_kernel()
    # Create a contiguous tensor and then make it non-contiguous by transposing.
    x = torch.randn(10, 20, device="cuda", dtype=torch.float32)
    x_noncontig = x.t()  # Transpose makes it non-contiguous
    with pytest.raises(RuntimeError, match="must be contiguous"):
        kernel.forward(x_noncontig, 0)

if __name__ == "__main__":
    pytest.main([__file__])
