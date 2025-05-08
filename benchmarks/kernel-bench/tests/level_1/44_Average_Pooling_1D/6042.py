
import pytest
import torch
from torch.utils.cpp_extension import load
import os

def build_kernel():
    cuda_module = load(
        name="avg_pool1d_ext",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Helper: define a simple python wrapper to mimic the nn.AvgPool1d behavior for comparison.
def avg_pool1d_reference(x, kernel_size, stride, padding):
    # x: (batch_size, in_channels, input_length)
    batch_size, in_channels, input_length = x.shape
    output_length = (input_length + 2 * padding - kernel_size) // stride + 1
    out = torch.empty((batch_size, in_channels, output_length), device=x.device, dtype=x.dtype)
    # pad input manually
    x_padded = torch.nn.functional.pad(x, (padding, padding), mode="constant", value=0)
    for b in range(batch_size):
        for c in range(in_channels):
            for i in range(output_length):
                start = i * stride
                end = start + kernel_size
                window = x_padded[b, c, start:end]
                # AvgPool1d in PyTorch (default) uses count_include_pad=True.
                out[b, c, i] = window.mean()
    return out

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestAvgPool1dKernel:
    def test_non_float_dtype(self):
        """
        Issue 1: When a non-float type (e.g. torch.double) is provided, the kernel
        (which expects float data) uses the pointer reinterpretation incorrectly.
        This test passes a double tensor and checks that the output is different
        from the reference computation done in double precision.
        """
        my_module = build_kernel()
        batch_size = 4
        in_channels = 3
        input_length = 20
        kernel_size = 3
        stride = 1
        padding = 1
        # Create a double tensor but send to CUDA.
        x = torch.randn(batch_size, in_channels, input_length, device="cuda", dtype=torch.double)
        # This will call avg_pool1d_forward_optimized which uses x.data_ptr<float>()
        # and thereby misinterpret the data.
        out_kernel = my_module.forward(x, kernel_size, stride, padding)
        out_ref = avg_pool1d_reference(x, kernel_size, stride, padding)
        # They should not match if the kernel misinterprets the data.
        with pytest.raises(AssertionError):
            assert torch.allclose(out_kernel.to(torch.double), out_ref, atol=1e-5)

    def test_non_contiguous_tensor(self):
        """
        Issue 2: The kernel expects a contiguous tensor. This test creates
        a non-contiguous tensor (by transposing the channels dimension) and ensures
        that misinterpretation of the data (or wrong result) occurs.
        """
        my_module = build_kernel()
        batch_size = 4
        in_channels = 4
        input_length = 30
        kernel_size = 3
        stride = 2
        padding = 1
        # Create a contiguous tensor in float first.
        x = torch.randn(batch_size, in_channels, input_length, device="cuda", dtype=torch.float32)
        # Make it non-contiguous by a permutation and then reverting the permutation partially:
        x_non_contig = x.transpose(1, 2)  # shape becomes (batch_size, input_length, in_channels)
        x_non_contig = x_non_contig.transpose(1, 2)  # back to (batch_size, in_channels, input_length) but non-contiguous
        assert not x_non_contig.is_contiguous(), "Tensor should be non-contiguous for this test"
        out_kernel = my_module.forward(x_non_contig, kernel_size, stride, padding)
        out_ref = avg_pool1d_reference(x_non_contig.contiguous(), kernel_size, stride, padding)
        # Expect a discrepancy (or an error) because the kernel did not account for non-contiguous layout.
        with pytest.raises(AssertionError):
            assert torch.allclose(out_kernel, out_ref, atol=1e-5)

    def test_missing_kernel_launch_error_check(self):
        """
        Issue 3: No error checking is done after kernel launch. We simulate an error
        by providing invalid kernel parameters (e.g., kernel_size=0) and expect the TORCH_CHECK
        in the forward wrapper to trigger.
        """
        my_module = build_kernel()
        batch_size = 2
        in_channels = 2
        input_length = 10
        kernel_size = 0  # invalid
        stride = 1
        padding = 0
        x = torch.randn(batch_size, in_channels, input_length, device="cuda", dtype=torch.float32)
        with pytest.raises(RuntimeError):
            _ = my_module.forward(x, kernel_size, stride, padding)

    def test_fixed_block_size_limitation(self):
        """
        Issue 4: The fixed block size of 256 may not handle very small output lengths gracefully.
        This test uses an input where output_length is less than the block size
        and verifies that the kernel output does not match the reference (i.e. the fixed block size
        strategy may lead to unused threads or misconfiguration).
        """
        my_module = build_kernel()
        batch_size = 2
        in_channels = 2
        input_length = 5  # small input length
        kernel_size = 3
        stride = 2
        padding = 1
        x = torch.randn(batch_size, in_channels, input_length, device="cuda", dtype=torch.float32)
        out_kernel = my_module.forward(x, kernel_size, stride, padding)
        out_ref = avg_pool1d_reference(x, kernel_size, stride, padding)
        # If the fixed block size is not adapted, the result may be incorrect.
        with pytest.raises(AssertionError):
            assert torch.allclose(out_kernel, out_ref, atol=1e-5)
