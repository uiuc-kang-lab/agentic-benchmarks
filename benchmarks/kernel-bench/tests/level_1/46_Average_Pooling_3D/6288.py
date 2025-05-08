
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build the CUDA extension from kernel.cu
def build_kernel():
    cuda_module = load(
        name="avg_pool3d_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Kernel only supports float32 input.
def test_data_type_support():
    cuda_module = build_kernel()
    # Create a double tensor which should not be supported by the kernel
    x = torch.randn(1, 1, 10, 10, 10, device="cuda", dtype=torch.float64)
    with pytest.raises(RuntimeError):
        # Expecting an error due to wrong data type.
        _ = cuda_module.forward(x, 3, 1, 0)

# Issue 2: Kernel assumes contiguous input.
def test_non_contiguous_input():
    cuda_module = build_kernel()
    # Create a contiguous tensor and a non-contiguous view of it.
    x = torch.randn(2, 4, 10, 10, 10, device="cuda", dtype=torch.float32)
    # Create non-contiguous input by transposing some dimensions
    x_non_contig = x.transpose(1, 2)
    # Compute pooling with PyTorch's built-in AvgPool3d (which works regardless)
    avg_pool_ref = torch.nn.AvgPool3d(kernel_size=3, stride=1, padding=1)
    y_ref = avg_pool_ref(x_non_contig.contiguous())  # reference requires contiguous memory

    # Using our CUDA kernel on non-contiguous input may result in unexpected results.
    y_kernel = cuda_module.forward(x_non_contig, 3, 1, 1)
    
    # The results are expected to differ because the kernel assumes contiguous memory.
    assert not torch.allclose(y_kernel, y_ref, atol=1e-4), \
        "Kernel produced correct results for non-contiguous input, but it was supposed to assume contiguity."

# Issue 3: Lack of support for extra pooling parameters (e.g. dilation, ceil_mode).
def test_advanced_pooling_parameters():
    cuda_module = build_kernel()
    # Create a sample input.
    x = torch.randn(1, 1, 20, 20, 20, device="cuda", dtype=torch.float32)
    # PyTorch's AvgPool3d with dilation (simulate advanced pooling). The extension does not support dilation.
    avg_pool_dilated = torch.nn.AvgPool3d(kernel_size=3, stride=2, padding=1, dilation=2)
    y_ref = avg_pool_dilated(x)
    # Our CUDA kernel always uses dilation=1 behavior.
    y_kernel = cuda_module.forward(x, 3, 2, 1)
    
    # In general the outputs should differ when dilation is used.
    assert not torch.allclose(y_kernel, y_ref, atol=1e-4), \
       "Kernel behaved as if dilation were supported, but it should not handle dilation > 1."

# Issue 4: Fixed unroll factor may not be appropriate when kernel_size < 4.
def test_unroll_factor_issue():
    cuda_module = build_kernel()
    # Use a kernel size smaller than the unroll factor (e.g., 2 < 4).
    x = torch.randn(1, 1, 8, 8, 8, device="cuda", dtype=torch.float32)
    avg_pool = torch.nn.AvgPool3d(kernel_size=2, stride=1, padding=0)
    y_ref = avg_pool(x)
    y_kernel = cuda_module.forward(x, 2, 1, 0)
    
    # We expect the results to be wrong because the fixed unroll is not adjusted for a small kernel.
    assert not torch.allclose(y_kernel, y_ref, atol=1e-4), \
        "Kernel produced correct results for kernel_size < unroll factor, but it was expected to trigger an issue."

# Issue 5: Potential overflow in index calculation for extremely large tensors.
@pytest.mark.skipif(True, reason="This test may require enormous memory and is meant to simulate index overflow.")
def test_index_overflow():
    cuda_module = build_kernel()
    # Intentionally create an input tensor with huge dimensions to simulate potential overflow.
    # Note: In practice this tensor is too large to allocate; this test is mainly illustrative.
    batch_size = 1
    channels = 1
    large_dim = 100000  # chosen to make the total number of elements extremely high (not allocatable in reality)
    try:
        x = torch.randn(batch_size, channels, large_dim, large_dim, large_dim, device="cuda", dtype=torch.float32)
    except RuntimeError:
        pytest.skip("Skipping test_index_overflow because memory allocation failed as expected for huge tensors.")
    
    # Compute pooling with some valid parameters; if overflow occurs internally the output will be wrong.
    y_kernel = cuda_module.forward(x, 3, 1, 1)
    # Since we cannot compute a valid reference result on such a huge tensor,
    # we simply check that the kernel did not crash and produced an output.
    assert y_kernel.numel() > 0, "Kernel produced an empty output due to overflow in index computation."

if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])
