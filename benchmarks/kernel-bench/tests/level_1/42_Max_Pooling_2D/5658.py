
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Helper function to compute reference max pool using PyTorch's max_pool2d.
def ref_maxpool2d(input, kernel_size, stride, padding, dilation):
    return torch.nn.functional.max_pool2d(input, kernel_size, stride, padding, dilation)

# Issue 1 Test: Non-contiguous input tensor.
def test_non_contiguous_input():
    # Create a contiguous tensor, then make it non-contiguous via a permute, then make it contiguous with wrong memory layout.
    batch_size, channels, height, width = 2, 3, 8, 8
    kernel_size = 2
    stride = 2
    padding = 1
    dilation = 1

    # Original contiguous input in NCHW.
    input_tensor = torch.randn(batch_size, channels, height, width, device="cuda", dtype=torch.float32)
    # Create a non-contiguous view by transposing height and width
    non_contig_input = input_tensor.permute(0, 1, 3, 2)  # now shape: [b, c, width, height]
    # Pass this non-contiguous tensor to the CUDA kernel.
    # The kernel assumes contiguous NCHW ordering, so the result is likely to be incorrect.
    kernel_module = build_kernel()
    try:
        output = kernel_module.forward(non_contig_input, kernel_size, stride, padding, dilation)
    except Exception as e:
        pytest.skip(f"Kernel raised an exception with non-contiguous input: {e}")
    # Compare against PyTorch's implementation (but with adjusted input shape).
    # Since the layout is wrong, the outputs should differ significantly.
    ref_output = ref_maxpool2d(non_contig_input, kernel_size, stride, padding, dilation)
    # We expect the output to be different from the reference due to layout mismatch.
    assert not torch.allclose(output, ref_output, atol=1e-5), (
        "Kernel did not fail for non-contiguous input; expected incorrect results."
    )

# Issue 2 Test: Using half precision input (float16).
def test_half_precision_input():
    batch_size, channels, height, width = 2, 3, 16, 16
    kernel_size = 3
    stride = 2
    padding = 1
    dilation = 1

    # Create half precision input.
    input_tensor = torch.randn(batch_size, channels, height, width, device="cuda", dtype=torch.float16)
    kernel_module = build_kernel()
    try:
        output = kernel_module.forward(input_tensor, kernel_size, stride, padding, dilation)
        # Use PyTorch’s own pooling for reference.
        ref_output = ref_maxpool2d(input_tensor, kernel_size, stride, padding, dilation)
    except Exception as e:
        pytest.skip(f"Kernel raised an exception for half-precision input: {e}")
    # There might be precision variations; check that differences are significant due to potential infinity issues.
    max_diff = (output - ref_output).abs().max().item()
    assert max_diff > 1e-2, (
        f"Kernel may not be handling half precision correctly: maximum difference {max_diff} too small."
    )
    
if __name__ == "__main__":
    pytest.main([__file__])
