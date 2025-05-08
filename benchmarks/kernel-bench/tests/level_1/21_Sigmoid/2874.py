
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_dtype_issue():
    """
    Test the kernel with a double precision tensor.
    The kernel is written for float32 and uses float4 vectorization and expf.
    Therefore, when given a double tensor, the computation will be incorrect.
    """
    module = build_kernel()
    # Create a tensor of type double.
    x = torch.randn(1024, device='cuda', dtype=torch.double)
    y = module.forward(x)
    y_ref = torch.sigmoid(x)
    # The kernel is not meant for double precision. Its result should be significantly off.
    assert not torch.allclose(y, y_ref, atol=1e-6), (
        "Kernel output is unexpectedly correct for double precision input despite "
        "the assumption of float32."
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tail_handling_issue():
    """
    Test the kernel with an input whose number of elements is not divisible by 4.
    The kernel’s vectorized loop will perform an out‐of‐bounds write in this case.
    """
    module = build_kernel()
    # Create an input of size that is not a multiple of 4.
    x = torch.randn(1023, device='cuda', dtype=torch.float32)
    # Running the kernel may corrupt out-of-bound memory.
    y = module.forward(x)
    y_ref = torch.sigmoid(x)
    # If the kernel mishandles the tail, the output will not match the reference result.
    assert not torch.allclose(y, y_ref, atol=1e-5), (
        "Kernel output matches reference even though the tail elements are not handled correctly."
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_alignment_issue():
    """
    Test the kernel with an input that is likely misaligned.
    By slicing a larger tensor, we force the data pointer of the resulting tensor to
    be misaligned with respect to the float4 boundary needed for vectorized accesses.
    """
    module = build_kernel()
    # Create a tensor larger than needed and then slice it to misalign the memory pointer.
    base = torch.randn(1024 + 1, device='cuda', dtype=torch.float32)
    x = base[1:]  # This slice is likely to be misaligned for float4 operations.
    y = module.forward(x)
    y_ref = torch.sigmoid(x)
    # With misaligned accesses, incorrect results are likely.
    assert not torch.allclose(y, y_ref, atol=1e-5), (
        "Kernel output is correct despite potential misalignment issues."
    )
