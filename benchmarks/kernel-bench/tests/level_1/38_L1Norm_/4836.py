
import torch
import pytest
from torch.utils.cpp_extension import load

# Helper function to compile and load the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="l1_norm_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True,
    )
    return cuda_module

def test_non_2d_input():
    # Issue 1: Test that passing a tensor that is not 2D triggers an error.
    # Create a 3D tensor, which the kernel does not support.
    x = torch.randn(4, 16, 8, device="cuda", dtype=torch.float32)
    my_module = build_kernel()
    with pytest.raises(RuntimeError) as excinfo:
        # Our kernel host function expects x.dim() == 2 and will trigger TORCH_CHECK failure.
        my_module.forward(x)
    assert "Expected 2D tensor" in str(excinfo.value)

def test_non_float32_input():
    # Issue 2: Test that passing a non-float32 tensor (e.g. float64) causes an error or misinterpretation.
    # The kernel casts the pointer to float*, so using a double tensor should lead to wrong results.
    x = torch.randn(16, 16384, device="cuda", dtype=torch.float64)
    my_module = build_kernel()
    # Since there is no explicit check for data type, the kernel will attempt to process the input.
    # We expect the output to differ substantially from correct normalization.
    out = my_module.forward(x.contiguous().to(torch.float32))
    # Now, intentionally feed the wrong type.
    with pytest.raises(RuntimeError):
        # This call is likely to produce wrong results or even memory errors.
        my_module.forward(x)

def test_misaligned_vectorized_input():
    # Issue 3: Test that a misaligned tensor triggers issues with vectorized loads.
    # We first create a tensor that is float32 and with D a multiple of 4.
    base = torch.randn(16, 16384 + 1, device="cuda", dtype=torch.float32)
    # Slicing the tensor to force misalignment along the inner dimension.
    # The resulting tensor has inner dimension 16384, but the data pointer may not be aligned to 16-bytes.
    x = base[:, 1:]
    my_module = build_kernel()
    # Even though D is divisible by 4, the misaligned pointer may lead to incorrect behavior in the vectorized path.
    out = my_module.forward(x)
    # Compare against a CPU reference computation (using the PyTorch implementation) that works correctly.
    # Note: Since the CUDA kernel might be misaligned, the normalized values might be incorrect.
    norm = torch.sum(torch.abs(x), dim=1, keepdim=True)
    ref = x / norm.clamp(min=1e-12)
    # Check that the maximum absolute difference is reasonable.
    # Here, we expect the failure to show as a significant difference between the kernel and reference.
    diff = (out - ref).abs().max().item()
    assert diff > 1e-3, f"Kernel unexpectedly produced correct results for misaligned input, diff={diff}"

def test_kernel_launch_error_checking():
    # Issue 4: Test that we can detect a kernel launch error.
    # We simulate an error by passing an input tensor with 0 columns.
    # This edge-case might cause invalid behavior in the kernel (e.g., an empty reduction).
    x = torch.randn(16, 0, device="cuda", dtype=torch.float32)
    my_module = build_kernel()
    with pytest.raises(RuntimeError):
        my_module.forward(x)
