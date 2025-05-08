
import torch
import pytest
from torch.utils.cpp_extension import load

# Utility function to build the CUDA extension module.
def build_kernel():
    cuda_module = load(
        name="maxpool_extension",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

# Issue 1 test: Pass a half precision tensor. This should fail because the kernel does not support float16.
def test_half_precision_not_supported():
    module = build_kernel()
    # Create an input tensor of shape (batch_size, channels, height, width) as half precision
    x = torch.randn(16, 32, 128, 128, device='cuda', dtype=torch.float16)
    kernel_size = 2
    stride = 2
    padding = 1
    dilation = 3
    with pytest.raises(RuntimeError):
        # The call should raise an error or exception because the dispatch macro does not support float16.
        module.forward(x, kernel_size, stride, padding, dilation)

# Issue 2 test: Pass a non-contiguous tensor.
def test_non_contiguous_input():
    module = build_kernel()
    # Create a contiguous tensor and then create a non-contiguous view by transposing dimensions.
    x = torch.randn(16, 32, 128, 128, device='cuda', dtype=torch.float32)
    x_non_contig = x.transpose(2, 3)  # This makes the tensor non-contiguous.
    
    # Both the optimized kernel and PyTorch's nn.MaxPool2d should ideally produce the same result if the kernel
    # handled non-contiguity properly. Here we expect a discrepancy.
    kernel_size = 2
    stride = 2
    padding = 1
    dilation = 3
    
    # Run the custom CUDA kernel.
    out_custom = module.forward(x_non_contig, kernel_size, stride, padding, dilation)
    
    # Run the reference PyTorch MaxPool2d (which requires contiguous input, so we manually make x contiguous for reference).
    pool = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    out_ref = pool(x_non_contig.contiguous())
    
    # Because the kernel does not check for contiguity, the custom version is likely to yield different results.
    # We assert that the outputs are not all close.
    assert not torch.allclose(out_custom, out_ref, atol=1e-5), \
        "Kernel should produce an incorrect result when provided a non-contiguous tensor."

# Issue 3 test: Lack of kernel launch error checking.
def test_kernel_launch_without_device_check():
    module = build_kernel()
    # Create an input tensor on CPU instead of CUDA.
    x_cpu = torch.randn(16, 32, 128, 128, device='cpu', dtype=torch.float32)
    kernel_size = 2
    stride = 2
    padding = 1
    dilation = 3
    # If a CPU tensor is provided while the kernel expects a CUDA pointer,
    # this should trigger a launch error. However, because the kernel does not check for errors,
    # the error might not be clearly reported.
    with pytest.raises(RuntimeError):
        module.forward(x_cpu, kernel_size, stride, padding, dilation)
        
if __name__ == "__main__":
    pytest.main([__file__])
