
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Assuming kernel.cu is in the same directory as this test file.
kernel_file = os.path.join(os.path.dirname(__file__), "kernel.cu")

def build_kernel():
    cuda_module = load(
        name="l2norm_module",
        sources=[kernel_file],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Insufficient tensor dimensions
def test_insufficient_dimensions():
    my_module = build_kernel()
    # Create a 1D tensor. The kernel expects at least 2 dimensions (batch, C)
    x = torch.randn(16384, device='cuda', dtype=torch.float32)
    with pytest.raises(IndexError):
        # Expect an IndexError (or similar) due to accessing size(1) on a 1D tensor.
        _ = my_module.forward(x)
        
# Issue 2: Hard-coded block configuration assumptions
def test_block_size_mismatch():
    # Although the kernel launch configuration is hard-coded inside the C++ code,
    # we can try to trigger the potential issue by forcing a different input size
    # that might cause some blocks to have fewer threads processing a large segment.
    my_module = build_kernel()
    # Use a tensor with more channels than elements_per_block such that the final block
    # for some vector processes only a few elements.
    batch_size = 4
    # Choose C such that C mod elements_per_block is a small number (e.g. 17 if elements_per_block is 2048).
    C = 2065  # 2048 + 17 to force a partial block at the end
    x = torch.randn(batch_size, C, device='cuda', dtype=torch.float32)
    
    # Although the kernel may run without throwing an error,
    # the accumulated sum may be slightly off because of the padded shared memory reductions.
    # To test this, compare with PyTorch's normalization.
    out_kernel = my_module.forward(x)
    norm = torch.norm(x, p=2, dim=1, keepdim=True)
    out_ref = x / (norm + 1e-12)  # add epsilon similarly to the kernel
    # Here we test that the error is within a loose tolerance.
    assert not torch.allclose(out_kernel, out_ref, atol=1e-7), "Kernel unexpectedly matches reference when block configuration assumptions are violated"

# Issue 3: Assumption that normalization occurs along dimension 1
def test_wrong_normalization_dimension():
    my_module = build_kernel()
    # Create a 3D tensor (e.g., batch, channels, height)
    # The kernel always uses dimension 1 (channels) as C. But user might expect normalization along a different dimension.
    batch_size, channels, height = 8, 16, 10
    x = torch.randn(batch_size, channels, height, device='cuda', dtype=torch.float32)
    
    # When calling our kernel, it will interpret dimension 1 (channels) as the normalization dimension,
    # while the user might want to normalize along, say, the last dimension.
    out_kernel = my_module.forward(x)
    # Compute reference normalization along a different dimension and expect a difference.
    norm_ref = torch.norm(x, p=2, dim=2, keepdim=True)  # normalizing along dim2 instead of dim1
    out_ref = x / (norm_ref + 1e-12)
    
    # The outputs should differ.
    assert not torch.allclose(out_kernel, out_ref, atol=1e-7), \
        "Kernel output unexpectedly matches reference output when normalization dim is mismatched"

# Issue 4: Inefficient memory accesses with non-contiguous inputs
def test_non_contiguous_input():
    my_module = build_kernel()
    # Create a tensor and make it non-contiguous by transposing or slicing.
    batch_size, C = 16, 16384
    x = torch.randn(C, batch_size, device='cuda', dtype=torch.float32').t()  # transpose to get non-contiguous memory
    # Ensure the tensor is non-contiguous.
    assert not x.is_contiguous(), "Tensor should be non-contiguous for this test."
    
    # Run the kernel. It is likely to produce the correct result but inefficient memory access via __ldg.
    out_kernel = my_module.forward(x)
    norm = torch.norm(x, p=2, dim=1, keepdim=True)
    out_ref = x / (norm + 1e-12)
    
    # The results should be close if computed correctly, however this test is meant to alert the developer.
    assert torch.allclose(out_kernel, out_ref, atol=1e-5), \
        "Kernel output does not match expected output for non-contiguous input; check memory alignment and __ldg usage"
