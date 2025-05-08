
import pytest
import torch
from torch.utils.cpp_extension import load

# Utility function to build and load the CUDA extension from kernel.cu.
def build_kernel():
    cuda_module = load(
        name="avg_pool3d_cuda",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# 1. Test that passing an input tensor of a type other than float32 produces an error.
def test_input_tensor_type():
    my_module = build_kernel()
    # Create a tensor of type double (not float32)
    x = torch.randn(16, 32, 64, 64, 64, dtype=torch.double, device="cuda")
    with pytest.raises(RuntimeError):
        # the kernel expects float32, so this should trigger an error or result in wrong behavior
        # Here we check for error, assuming that proper type checking would be added.
        out = my_module.forward(x, 3, 2, 1)
        # Synchronize to catch any asynchronous error.
        torch.cuda.synchronize()

# 2. Test that non-cubic pooling parameters (simulated by passing a non-contiguous tensor)
#    lead to incorrect results if the input layout is not as expected.
def test_non_contiguous_input():
    my_module = build_kernel()
    # Create a contiguous tensor of shape (batch, channels, depth, height, width)
    x = torch.randn(16, 32, 64, 64, 64, device="cuda", dtype=torch.float32)
    # Make the tensor non-contiguous by a transpose operation
    x_non_contig = x.transpose(1, 4)  # this will change the memory layout
    # The kernel assumes NCDHW layout, so comparing the kernel result on a non-contiguous tensor with nn.AvgPool3d
    # (after making it contiguous) will reveal an inconsistency.
    out_cuda = my_module.forward(x_non_contig, 3, 2, 1)
    # Compute reference output using PyTorch's native AvgPool3d
    avgpool = torch.nn.AvgPool3d(kernel_size=3, stride=2, padding=1)
    # Ensure the reference input is contiguous
    out_ref = avgpool(x_non_contig.contiguous())
    # The outputs are likely to differ because of incorrect memory assumptions.
    with pytest.raises(AssertionError):
        assert torch.allclose(out_cuda, out_ref, atol=1e-5), "Non-contiguous input not handled correctly."

# 3. Test that using non-cubic kernel parameters reveals the limitation of the kernel.
#    Although the extension interface accepts only one kernel_size value, we simulate the scenario by
#    comparing the output to nn.AvgPool3d when the pooling window would be non-cubic at borders.
def test_non_cubic_border_effect():
    my_module = build_kernel()
    # Create an input tensor with spatial dimensions where the border pooling window will be truncated 
    # in one or more dimensions due to padding.
    # For example, use a smaller depth than expected.
    x = torch.randn(16, 32, 5, 64, 64, device="cuda", dtype=torch.float32)
    out_cuda = my_module.forward(x, 3, 2, 1)
    avgpool = torch.nn.AvgPool3d(kernel_size=3, stride=2, padding=1)  # count_include_pad is True
    out_ref = avgpool(x)
    # Since the custom CUDA kernel always divides by a full kernel volume, the border behavior might differ.
    # We expect a mismatch.
    with pytest.raises(AssertionError):
        assert torch.allclose(out_cuda, out_ref, atol=1e-5), "Border pooling window handled incorrectly for non-cubic effect."

# 4. There is no direct runtime test for the potential namespace/min-max issues in the kernel since they are compile‐time problems.
#    However, attempting to build the module itself (in build_kernel) should surface any such issues.
def test_kernel_build():
    # This test will pass if the kernel compiles successfully.
    try:
        my_module = build_kernel()
    except Exception as e:
        pytest.fail(f"Kernel build failed due to min/max namespace issues or other compile errors: {e}")
