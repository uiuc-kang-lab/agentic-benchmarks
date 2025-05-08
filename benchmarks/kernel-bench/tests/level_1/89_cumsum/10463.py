
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Helper to build and load the CUDA extension from "kernel.cu"
def build_kernel():
    module = load(
        name="test_module",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return module

# Issue 1: Conditional __syncthreads() usage – this may lead to undefined behavior.
# We design a test that uses an input where the scan dimension is large enough that some warps enter the loop and call __syncthreads()
# while others do not. This will likely yield an incorrect result compared to torch.cumsum.
def test_conditional_sync():
    torch.manual_seed(0)
    # Use a shape with scan dimension size not a multiple of 32 and large enough to trigger multiple iterations.
    # For example, shape (4, 97) on CUDA.
    x = torch.randn(4, 97, device="cuda", dtype=torch.float32).contiguous()
    module = build_kernel()
    # Our kernel always scans along dim 1
    kernel_result = module.forward(x, 1)
    ref_result = torch.cumsum(x, dim=1)
    # We expect a discrepancy due to synchronization bug.
    # Using a loose tolerance, but the results likely differ significantly.
    assert not torch.allclose(kernel_result, ref_result, atol=1e-5), (
        "Conditional __syncthreads usage issue not triggered: kernel output unexpectedly matches torch.cumsum."
    )

# Issue 2: Kernel only handles float32 – using a different data type (double) should produce wrong results.
def test_data_type_support():
    torch.manual_seed(0)
    x = torch.randn(8, 64, device="cuda", dtype=torch.float64).contiguous()  # double precision
    module = build_kernel()
    # Although our kernel does not check for data type, it uses data_ptr<float>().
    kernel_result = module.forward(x, 1)
    ref_result = torch.cumsum(x, dim=1)
    # Since the kernel misinterprets data the output should differ significantly.
    assert not torch.allclose(kernel_result, ref_result, atol=1e-5), (
        "Data type support issue not triggered: kernel output for double tensor matches torch.cumsum."
    )

# Issue 3: The kernel assumes the cumulative sum dimension is contiguous.
# Passing a noncontiguous tensor (with respect to the scanned dimension) should trigger the CHECK_CONTIGUOUS
# and raise an error.
def test_non_contiguous_input():
    torch.manual_seed(0)
    # Create a tensor and then permute it so that the dimension to be scanned (say, dim=1)
    # is not contiguous.
    x = torch.randn(16, 32, 8, device="cuda", dtype=torch.float32)
    x_noncontig = x.permute(0, 2, 1).contiguous(memory_format=torch.channels_last)  # force a non-standard layout
    # Make sure the tensor is noncontiguous
    assert not x_noncontig.is_contiguous(), "The test tensor is unexpectedly contiguous."
    module = build_kernel()
    with pytest.raises(RuntimeError, match="must be contiguous"):
        module.forward(x_noncontig, 2)

# Issue 4: Work distribution assumptions may break when the scan dimension size
# is not a multiple of 32. This can lead to incorrect accumulation over warps.
def test_scan_dim_not_multiple_of_32():
    torch.manual_seed(0)
    # Use a scan dimension (dim=1) whose size is not a multiple of 32.
    x = torch.randn(10, 45, device="cuda", dtype=torch.float32).contiguous()
    module = build_kernel()
    kernel_result = module.forward(x, 1)
    ref_result = torch.cumsum(x, dim=1)
    # Expect the result to differ due to misaccumulation from improper warp handling.
    assert not torch.allclose(kernel_result, ref_result, atol=1e-5), (
        "Work distribution issue not triggered: kernel output for non-multiple-of-32 scan dimension matches torch.cumsum."
    )
