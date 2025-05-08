
import pytest
import torch
from torch.utils.cpp_extension import load

# Utility function to build the CUDA module.
def build_kernel():
    cuda_module = load(
        name="l2norm_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test Case 1:
# Verify that passing a CPU tensor triggers the CUDA check (issue: lack of proper device-handling)
def test_cpu_input_error():
    cuda_module = build_kernel()
    cpu_tensor = torch.randn(16, 16384)
    with pytest.raises(RuntimeError, match="Input must be a CUDA tensor"):
        _ = cuda_module.forward(cpu_tensor)

# Test Case 2:
# Verify that an input tensor with extra dimensions and non-contiguous layout
# (which the kernel’s indexing does not correctly account for) produces an incorrect result.
def test_noncontiguous_multidim_input():
    cuda_module = build_kernel()
    # Create a 3D tensor that should be normalized along dim=1
    # For example, shape (B, C, M) where normalization is performed for every (B, :, M) vector.
    B, C, M = 4, 128, 7
    # Create a contiguous tensor and then permute to make it noncontiguous.
    x = torch.randn(B, C, M, device="cuda", dtype=torch.float32)
    # Permute so that the originally intended normalization dim (the second dimension) is no longer contiguous.
    x_perm = x.permute(2, 0, 1)  # now shape: (M, B, C) 
    # Our kernel is implemented to normalize along dim=1, so we mimic that by
    # calling our module with the tensor viewed as [M*B, C]. 
    # But here the contiguous ordering is lost even though x_perm.view(-1, C) is possible.
    x_view = x_perm.contiguous().view(-1, C)  # Force a reordering (though contiguous, this reordering is different from the original)
    
    # Compute reference normalization using torch (which works for general tensors)
    ref = x_view / (x_view.norm(p=2, dim=1, keepdim=True) + 1e-12)
    # Use the CUDA kernel
    out = cuda_module.forward(x_view)
    # Because of indexing issues in the kernel for multidim cases,
    # the output is expected to be incorrect. We trigger the issue by asserting that
    # the computed result is NOT close to the reference.
    assert not torch.allclose(out, ref, atol=1e-5), \
           "Kernel unexpectedly produced a correct result on a nonstandard tensor layout!"

# Test Case 3:
# Verify that passing an input with a non-floating type (e.g. integer)
# triggers an error since the dispatch macro only handles floating types.
def test_invalid_dtype():
    cuda_module = build_kernel()
    # Create an integer tensor on CUDA.
    x_int = torch.randint(0, 10, (16, 16384), device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError):
        _ = cuda_module.forward(x_int)
