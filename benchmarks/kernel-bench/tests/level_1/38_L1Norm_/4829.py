
import pytest
import torch
from torch.utils.cpp_extension import load
import numpy as np

# Helper function to build the CUDA extension from kernel.cu
def build_kernel():
    return load(
        name="l1_norm_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )

# A helper function to compute L1 normalization on CPU (mimicking the PyTorch version)
def torch_l1_norm(x: torch.Tensor, dim: int = 1):
    # The PyTorch function does not add an epsilon. Dividing by 0 yields nan.
    s = torch.sum(torch.abs(x), dim=dim, keepdim=True)
    return x / s

# Test case for Issue 1: Kernel only supports 2D inputs.
def test_kernel_non_2d_input():
    cuda_module = build_kernel()
    # Create a 3D tensor, while the kernel expects a 2D tensor.
    x = torch.randn(4, 16, 32, device="cuda")
    # We expect our host launch to trigger a check from the kernel code.
    with pytest.raises(RuntimeError, match="Expected 2D tensor"):
        # Our host function in kernel.cu uses: TORCH_CHECK(x.dim() == 2, "Expected 2D tensor for this example.");
        cuda_module.forward(x)

# Test case for Issue 2: Kernel implemented only for float32.
def test_kernel_incorrect_dtype():
    cuda_module = build_kernel()
    # Create a 2D tensor with dtype float64; kernel expects float (float32).
    x = torch.randn(16, 16384, device="cuda", dtype=torch.float64)
    # Since the kernel does not implement conversions, the result will be miscomputed.
    # We do not necessarily get a MAC error but the output will not be correct.
    out = cuda_module.forward(x)
    # Compute the reference result using PyTorch separated by converting x to float32 for the kernel to work properly.
    x_float = x.to(torch.float32)
    ref = torch_l1_norm(x_float, dim=1)
    # The output from kernel will be computed on the float32 view of the underlying storage.
    # They likely will not match because of the conversion issue.
    max_diff = (out - ref).abs().max().item()
    assert max_diff > 1e-3, f"Kernel unexpectedly produced similar output for wrong dtype; max diff: {max_diff}"

# Test case for Issue 3: Divergent behavior in zero rows.
def test_kernel_zero_row_behavior():
    cuda_module = build_kernel()
    # Create an input tensor where one row is all zeros.
    x = torch.randn(16, 16384, device="cuda", dtype=torch.float32)
    x[0] = 0.0  # first row all zeros
    out_kernel = cuda_module.forward(x)
    # Compute reference using PyTorch division (which leads to division by zero, i.e. nan)
    out_ref = torch_l1_norm(x, dim=1)
    # For the zero row, torch division yields nan whereas our kernel substitutes 1e-12 enabling division (0/1e-12 = 0)
    # Therefore, we expect the first row from the kernel to be all 0's while the reference is all nan.
    kernel_row = out_kernel[0]
    ref_row = out_ref[0]
    # Check that reference row is nan and kernel row is 0.
    assert torch.isnan(ref_row).all(), "Reference computation for zero row should result in NaNs."
    assert torch.equal(kernel_row, torch.zeros_like(kernel_row)), "Kernel did not produce zeros on zero input row."

# Test case for Issue 4: Lack of error checking on CUDA kernel launch.
# Although this issue is not directly triggerable via input data, we can simulate a launch error
# by passing an input tensor that is not on CUDA. This should trigger a TORCH_CHECK in the host code.
def test_kernel_input_not_on_cuda():
    cuda_module = build_kernel()
    # Create a tensor on CPU
    x = torch.randn(16, 16384, device="cpu", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="Input tensor must be on CUDA."):
        cuda_module.forward(x)
