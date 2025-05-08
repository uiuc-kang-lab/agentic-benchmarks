
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="leaky_relu_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_input_type_mismatch():
    # Issue 1: The kernel only supports float32 tensors.
    # Create a tensor of type float64 and expect the CHECK_CUDA macro to pass,
    # but the kernel will be launched with wrong type underneath.
    x = torch.randn(1024, 1024, dtype=torch.double, device="cuda")
    kernel_module = build_kernel()
    with pytest.raises(RuntimeError):
        # Expecting a runtime error due to type incompatibility when accessing data_ptr<float>()
        kernel_module.forward(x, 0.01)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_non_contiguous_tensor():
    # Issue 1 (extended): The kernel checks that the tensor is contiguous.
    # Create a non-contiguous tensor.
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)
    x_non_contig = x.t()  # Transpose creates a non-contiguous tensor.
    kernel_module = build_kernel()
    with pytest.raises(RuntimeError):
        kernel_module.forward(x_non_contig, 0.01)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_negative_slope_greater_than_one():
    # Issue 2: For negative_slope > 1, the kernel uses fmaxf which gives incorrect results.
    # We compare the results of our kernel with the expected PyTorch behavior (element-wise).
    negative_slope = 1.5  # > 1 should trigger the fmaxf issue
    kernel_module = build_kernel()
    x = torch.randn(1024, device="cuda", dtype=torch.float32)
    # Compute with the custom CUDA kernel:
    out_kernel = kernel_module.forward(x, negative_slope)
    # Compute with PyTorch reference implementation:
    out_ref = torch.where(x >= 0, x, x * negative_slope)
    # The kernel is expected to be wrong for some values.
    # We check if there is any discrepancy.
    if torch.allclose(out_kernel, out_ref, atol=1e-6):
        pytest.fail("Kernel produced correct results with negative_slope > 1, but an error was expected.")
    else:
        # Optionally, print max difference for debugging.
        diff = (out_kernel - out_ref).abs().max().item()
        print(f"Max difference with negative_slope > 1: {diff}")

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_missing_kernel_launch_error_check():
    # Issue 3: There is no error checking after the kernel launch. A misconfiguration (e.g. having a very large
    # tensor that exceeds GPU limits) might fail silently.
    # We simulate a situation that might trigger a launch error by using an extremely large array.
    kernel_module = build_kernel()
    # A tensor with an enormous number of elements
    try:
        # We use a shape that likely cannot be allocated on typical GPUs.
        x = torch.empty(int(1e10), device="cuda", dtype=torch.float32)
    except RuntimeError:
        pytest.skip("Skipping test because requested allocation could not be performed on the device")
    with pytest.raises(RuntimeError):
        # Expect that a kernel launch error will be raised or the runtime will signal an error.
        kernel_module.forward(x, 0.01)
