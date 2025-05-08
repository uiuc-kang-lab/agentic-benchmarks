
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="elu_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )
    return cuda_module

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_wrong_dtype():
    """
    Test that passing a tensor of the wrong dtype (e.g., float64 instead of float32)
    causes incorrect results because the kernel does not check for dtype.
    """
    module = build_kernel()
    # Create a double tensor on CUDA.
    x = torch.randn(1024, device="cuda", dtype=torch.float64)
    # Run the custom cuda kernel (this will reinterpret the double bits as float)
    y_kernel = module.forward(x, 1.0)
    # Compute PyTorch ELU using float64; values will be different.
    y_pytorch = F.elu(x, alpha=1.0)
    # They should not match because the kernel misinterprets the underlying data.
    assert not torch.allclose(y_kernel.to(torch.float64), y_pytorch, atol=1e-5), \
        "Kernel unexpectedly produced correct results for wrong dtype input."

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_non_contiguous():
    """
    Test that a non‑contiguous input tensor is rejected.
    """
    module = build_kernel()
    # Create a tensor and make it non-contiguous using a transpose.
    x = torch.randn(32, 32, device="cuda", dtype=torch.float32)
    x_noncontig = x.t()  # Transposed tensor is typically non-contiguous.
    with pytest.raises(RuntimeError, match="must be contiguous"):
        module.forward(x_noncontig, 1.0)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_misaligned_memory():
    """
    Test that input tensor with misaligned memory (not aligned to 16 bytes)
    produces incorrect results due to the unsafe reinterpret_cast to float4.
    Note: Since many CUDA devices support unaligned accesses without crashing,
    we compare against PyTorch's ELU and expect a discrepancy in the result.
    """
    module = build_kernel()
    # Allocate a tensor with extra element and then slice it so that the data pointer is offset.
    total = 1024
    base = torch.empty(total + 1, device="cuda", dtype=torch.float32)
    # Slicing from index 1 may produce a misaligned pointer.
    x = base[1:total+1].contiguous()
    
    # Check pointer alignment; if unexpectedly aligned, skip the test.
    ptr = x.data_ptr()
    if ptr % 16 == 0:
        pytest.skip("Tensor memory is aligned; cannot trigger misaligned access scenario.")
    
    y_kernel = module.forward(x, 1.0)
    y_pytorch = F.elu(x, alpha=1.0)
    # Expect that the results differ enough to signal an error due to misaligned memory.
    assert not torch.allclose(y_kernel, y_pytorch, atol=1e-4), \
        "Kernel output unexpectedly correct for misaligned memory input."
