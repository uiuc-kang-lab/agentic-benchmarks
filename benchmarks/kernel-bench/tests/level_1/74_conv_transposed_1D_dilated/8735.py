
import torch
import pytest
from torch.utils.cpp_extension import load
import os

# Build the CUDA extension from kernel.cu
def build_kernel():
    # Ensure the file exists in the same directory as the test file or adjust path accordingly.
    kernel_path = os.path.join(os.path.dirname(__file__), "kernel.cu")
    cuda_module = load(
        name="test_kernel",
        sources=[kernel_path],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test case 1: Trigger issue with unsupported data type (double instead of float)
def test_unsupported_dtype():
    # Prepare input tensors with double precision.
    batch_size = 2
    in_channels = 3
    length = 16
    out_channels = 4
    kernel_size = 3
    stride = 1
    padding = 0
    dilation = 1

    # Create tensors on CUDA but of type double
    x = torch.randn(batch_size, in_channels, length, device='cuda', dtype=torch.double)
    # Weight tensor shape: [C_in, C_out, kernel_size]
    weight = torch.randn(in_channels, out_channels, kernel_size, device='cuda', dtype=torch.double)
    bias = torch.randn(out_channels, device='cuda', dtype=torch.double)

    module = build_kernel()
    # This call should trigger an error or produce wrong results since the kernel expects float32.
    with pytest.raises(AssertionError) as excinfo:
        module.forward(x, weight, bias, stride, padding, dilation)
    assert "CUDA kernel failed" in str(excinfo.value) or "assert" in str(excinfo.value)

# Test case 2: Trigger issue with non-contiguous tensors.
def test_non_contiguous_tensors():
    batch_size = 2
    in_channels = 3
    length = 16
    out_channels = 4
    kernel_size = 3
    stride = 1
    padding = 0
    dilation = 1

    # Create contiguous tensors and then make them non-contiguous by transposing dimensions.
    x = torch.randn(batch_size, length, in_channels, device='cuda', dtype=torch.float32).transpose(1, 2)
    weight = torch.randn(out_channels, kernel_size, in_channels, device='cuda', dtype=torch.float32)  # Wrong layout
    # Make weight non-contiguous by a permutation that does not match expected layout ([C_in, C_out, K])
    weight = weight.permute(2, 0, 1)  # Now weight is of shape [in_channels, out_channels, kernel_size] but non-contiguous.
    bias = torch.randn(out_channels, device='cuda', dtype=torch.float32)

    module = build_kernel()
    # The forward call in the kernel binding calls .contiguous() on the input,
    # so the non-contiguity may be hidden on the input x but the test is designed to show that
    # not checking layout can lead to issues when layout is not as expected.
    y = module.forward(x, weight, bias, stride, padding, dilation)
    # We don't have a runtime correctness oracle, so we check that the output is not all zeros,
    # hinting potential mis-read data.
    assert y.abs().sum().item() != 0, "Kernel produced an output that is unexpectedly all zeros."

# Test case 3: Trigger issue with excessive shared memory usage.
def test_excessive_shared_memory():
    # Create a weight tensor that is very large so that the dynamic shared memory allocation
    # exceeds the hardware limit.
    batch_size = 1
    in_channels = 256
    out_channels = 256
    length = 32
    kernel_size = 11  # Larger kernel size increases weight memory footprint
    stride = 1
    padding = 0
    dilation = 1

    x = torch.randn(batch_size, in_channels, length, device='cuda', dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels, kernel_size, device='cuda', dtype=torch.float32)
    bias = torch.randn(out_channels, device='cuda', dtype=torch.float32)

    module = build_kernel()
    # Expected to fail due to exceeding shared memory limits.
    with pytest.raises(Exception) as excinfo:
        y = module.forward(x, weight, bias, stride, padding, dilation)
        # Synchronize to force kernel launch errors.
        torch.cuda.synchronize()
    assert "CUDA kernel failed" in str(excinfo.value) or "launch" in str(excinfo.value)

# Test case 4: Trigger potential performance degradation due to thread divergence.
def test_thread_divergence():
    # While we cannot easily measure divergence from a functional test,
    # we craft parameters with stride > 1 and dilation > 1 to force many threads to hit the
    # "if (l_in_nom % stride != 0) continue;" condition.
    batch_size = 2
    in_channels = 3
    length = 32
    out_channels = 4
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 2

    x = torch.randn(batch_size, in_channels, length, device='cuda', dtype=torch.float32)
    weight = torch.randn(in_channels, out_channels, kernel_size, device='cuda', dtype=torch.float32)
    bias = torch.randn(out_channels, device='cuda', dtype=torch.float32)

    module = build_kernel()
    y = module.forward(x, weight, bias, stride, padding, dilation)
    # We can at least check that the output tensor has the expected dimensions.
    # Compute expected L_out = (L_in - 1) * stride - 2 * padding + dilation * (K_w - 1) + 1
    L_out_expected = (length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    assert y.shape == (batch_size, out_channels, L_out_expected), f"Output shape mismatch. Got {y.shape}, expected {(batch_size, out_channels, L_out_expected)}"
