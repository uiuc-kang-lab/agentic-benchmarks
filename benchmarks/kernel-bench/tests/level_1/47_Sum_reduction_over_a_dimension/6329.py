
import pytest
import torch
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

# Issue 1: Non-contiguous input tensor
def test_non_contiguous_input():
    # Create a contiguous tensor and then take a transpose to get a non-contiguous layout.
    # Use a 3D tensor where reduction is over dim=1.
    batch_size, dim1, dim2 = 4, 8, 8
    original = torch.randn(batch_size, dim1, dim2, device='cuda')
    # Transpose to produce a non-contiguous tensor.
    non_contig = original.transpose(1, 2)
    # Since our kernel assumes contiguous memory with [outer, reduce, inner] pattern,
    # we need to simulate reduction along a dimension. We pick dim=1 (after transpose, this is not contiguous).
    reduce_dim = 1
    kernel_module = build_kernel()
    with pytest.raises(AssertionError):
        # The kernel will compute wrong results or may silently give erroneous results.
        # We trigger the issue by comparing against torch.sum, expecting a mismatch.
        out_cuda = kernel_module.forward(non_contig, reduce_dim)
        torch.cuda.synchronize()
        out_torch = torch.sum(non_contig, dim=reduce_dim, keepdim=True)
        # We force a test failure if they are (incorrectly) equal.
        assert not torch.allclose(out_cuda, out_torch), "Kernel incorrectly handled non-contiguous tensor"
        
# Issue 2: Lack of kernel launch error checking
def test_kernel_launch_failure():
    # Force a kernel launch error by supplying an input that is on CPU instead of CUDA.
    batch_size, dim1, dim2 = 4, 8, 8
    cpu_tensor = torch.randn(batch_size, dim1, dim2, device='cpu')
    reduce_dim = 1
    kernel_module = build_kernel()
    with pytest.raises(RuntimeError):
        # Using a CPU tensor should trigger CUDA launch issues.
        kernel_module.forward(cpu_tensor, reduce_dim)
    # Note: Depending on system setup, this may cause a runtime error from the CUDA API.
    
# Issue 3: Limited data type support due to dispatch macro
def test_input_tensor_non_floating():
    # Create an integer tensor which is not covered by AT_DISPATCH_FLOATING_TYPES.
    batch_size, dim1, dim2 = 4, 8, 8
    int_tensor = torch.randint(low=0, high=10, size=(batch_size, dim1, dim2), device='cuda', dtype=torch.int32)
    reduce_dim = 1
    kernel_module = build_kernel()
    
    with pytest.raises(RuntimeError):
        # The dispatch macro should raise an error for unsupported types such as int32.
        kernel_module.forward(int_tensor, reduce_dim)
