
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="avg_pool3d_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_large_grid_dimension():
    # Issue 1: Trigger grid.z limits by packing too many indices.
    # Here, batch=128, channels=128, and a shallow depth leads to grid.z = 128 * 128 * out_d.
    # For kernel_size=3, stride=2, padding=1, choose in_d=7 gives out_d = 4.
    # grid.z = 128 * 128 * 4 = 65536, which exceeds the typical maximum of 65535.
    batch_size = 128
    channels = 128
    in_d = 7  # yields out_d would be 4
    in_h = 16
    in_w = 16
    kernel_size = 3
    stride = 2
    padding = 1
    x = torch.randn(batch_size, channels, in_d, in_h, in_w, device="cuda", dtype=torch.float32)
    cuda_module = build_kernel()
    with pytest.raises(RuntimeError):
        out = cuda_module.forward(x, kernel_size, stride, padding)
        torch.cuda.synchronize()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_dtype():
    # Issue 2: The kernel only supports float32.
    # Create an input tensor with dtype float64 and expect the kernel to fail.
    batch_size = 2
    channels = 2
    in_d = 8
    in_h = 8
    in_w = 8
    kernel_size = 3
    stride = 2
    padding = 1
    x = torch.randn(batch_size, channels, in_d, in_h, in_w, device="cuda", dtype=torch.float64)
    cuda_module = build_kernel()
    with pytest.raises(RuntimeError):
        out = cuda_module.forward(x, kernel_size, stride, padding)
        torch.cuda.synchronize()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_count_include_pad_issue():
    # Issue 3: The kernel always divides by the full pooling volume.
    # For an input with ones and padding at the border, compare the kernel's output (which uses a fixed divisor)
    # with PyTorch's AvgPool3d configured with count_include_pad=False.
    batch_size = 1
    channels = 1
    in_d = 5
    in_h = 5
    in_w = 5
    kernel_size = 3
    stride = 1
    padding = 1
    x = torch.ones(batch_size, channels, in_d, in_h, in_w, device="cuda", dtype=torch.float32)
    cuda_module = build_kernel()
    out_cuda = cuda_module.forward(x, kernel_size, stride, padding)
    # PyTorch's AvgPool3d (when count_include_pad is False) will compute a higher average along borders.
    avg_pool = torch.nn.AvgPool3d(kernel_size=kernel_size, stride=stride, padding=padding, count_include_pad=False)
    out_torch = avg_pool(x)
    # They should differ because our kernel rigidly uses kernel_size^3 as the divisor.
    assert not torch.allclose(out_cuda, out_torch, atol=1e-5), \
        "Kernel incorrectly handles the count_include_pad option!"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_non_contiguous_input():
    # Issue 4: The kernel assumes a contiguous input.
    # Create a non-contiguous tensor (by permuting dimensions) and compare the result
    # to the same tensor after making it contiguous. The results are expected to differ.
    batch_size = 2
    channels = 3
    in_d = 8
    in_h = 8
    in_w = 8
    kernel_size = 3
    stride = 2
    padding = 1
    x = torch.randn(batch_size, channels, in_d, in_h, in_w, device="cuda", dtype=torch.float32)
    # Create a non-contiguous tensor by permuting dimensions.
    x_non_contig = x.permute(0, 2, 1, 3, 4)
    cuda_module = build_kernel()
    out_cuda = cuda_module.forward(x_non_contig, kernel_size, stride, padding)
    # Get the reference by making the input contiguous.
    out_reference = cuda_module.forward(x_non_contig.contiguous(), kernel_size, stride, padding)
    # The outputs should differ if the kernel does not handle non-contiguous memory properly.
    assert not torch.allclose(out_cuda, out_reference, atol=1e-5), \
        "Kernel does not exhibit the expected issue with non-contiguous inputs!"
