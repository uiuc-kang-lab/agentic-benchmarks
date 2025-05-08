
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    # Build the kernel from kernel.cu. Adjust extra_cflags as needed.
    cuda_module = load(
        name="sigmoid_kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

@pytest.fixture(scope='module')
def kernel_module():
    return build_kernel()

def test_double_type(kernel_module):
    # Issue 1: The kernel is not handling non-float32 types correctly.
    # Create a double tensor input.
    x = torch.randn(1024, device="cuda", dtype=torch.double)
    
    # Compute reference using torch.sigmoid
    ref = torch.sigmoid(x)
    
    # Call the custom CUDA kernel (which is dispatched with AT_DISPATCH_FLOATING_TYPES,
    # but inside always uses expf and static_cast to float, which is not precise for double).
    out = kernel_module.forward(x)
    torch.cuda.synchronize()
    
    # The outputs are expected to be noticeably different when using double
    # because of precision loss and wrong math calls.
    # We assert that the maximum difference is significant.
    max_diff = (out - ref).abs().max().item()
    assert max_diff > 1e-3, f"Expected noticeable error for double tensor, but max diff was {max_diff}"

def test_non_contiguous_input(kernel_module):
    # Issue 2: The kernel assumes contiguous memory, but many PyTorch operations yield non-contiguous tensors.
    # Create a contiguous tensor and then generate a non-contiguous view by transposing.
    x_contig = torch.randn(128, 256, device="cuda", dtype=torch.float32)
    x = x_contig.t()  # Transpose: now x is non-contiguous
    
    # Compute reference using torch.sigmoid (which correctly handles non-contiguous tensors).
    ref = torch.sigmoid(x)
    
    # Call the custom CUDA kernel.
    out = kernel_module.forward(x)
    torch.cuda.synchronize()
    
    # Due to kernel assuming contiguous layout, the result will differ from the reference.
    max_diff = (out - ref).abs().max().item()
    assert max_diff > 1e-3, f"Expected error for non-contiguous tensor, but max diff was {max_diff}"
