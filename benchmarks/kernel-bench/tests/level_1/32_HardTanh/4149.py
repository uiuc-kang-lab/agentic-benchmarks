
import pytest
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

# Build the CUDA extension from kernel.cu
def build_kernel():
    return load(
        name="hardtanh_extension",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=True
    )

# Helper: Reference HardTanh using PyTorch's function.
def ref_hardtanh(x, min_val, max_val):
    return F.hardtanh(x, min_val=min_val, max_val=max_val)

# Issue 1: Misaligned memory due to offset (alignment issue)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_misaligned_memory():
    torch.manual_seed(0)
    # Allocate a slightly larger tensor
    base = torch.randn(1025, device='cuda', dtype=torch.float32)
    # Create a misaligned tensor by slicing off one element.
    x = base.narrow(0, 1, 1024).clone()  # clone to ensure a new allocation with unknown alignment
    # Ensure x is contiguous but possibly misaligned relative to the allocation base.
    x = x.contiguous()
    min_val = -1.0
    max_val = 1.0

    kernel = build_kernel()
    out_kernel = kernel.forward(x, min_val, max_val)
    out_ref = ref_hardtanh(x, min_val, max_val)
    # If misalignment causes an issue, the kernel output may differ from the reference.
    assert torch.allclose(out_kernel, out_ref, atol=1e-5), "Kernel output differs on misaligned memory input!"

# Issue 2: Non-contiguous tensor input causes incorrect behavior.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_non_contiguous_input():
    torch.manual_seed(0)
    # Create a contiguous 2D tensor
    x = torch.randn(32, 64, device='cuda', dtype=torch.float32)
    # Make it non-contiguous by transposing.
    x_noncontig = x.t()
    min_val = -1.0
    max_val = 1.0

    kernel = build_kernel()
    with pytest.raises(Exception):
        # Expect the kernel to produce an error or wrong result since reinterpret_cast is invalid for non-contiguous data.
        # (In a robust implementation one would check contiguity and maybe fall back to a non-vectorized version.)
        _ = kernel.forward(x_noncontig, min_val, max_val)

# Issue 3: Lack of optimized support for half precision.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_half_precision():
    torch.manual_seed(0)
    x = torch.randn(1024, device='cuda', dtype=torch.float16)
    min_val = -1.0
    max_val = 1.0

    kernel = build_kernel()
    # Even if the kernel falls back to scalar processing, the output should be correct.
    out_kernel = kernel.forward(x, min_val, max_val)
    out_ref = ref_hardtanh(x, min_val, max_val)
    # Here we check correctness; however, note the lack of vectorized optimization.
    assert torch.allclose(out_kernel, out_ref, atol=1e-2), "Kernel output differs for half precision input!"

# Extra test: Input length not divisible by the vectorization width.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_non_divisible_numel():
    torch.manual_seed(0)
    # For float32, expected vector width is 4. Here we create an input whose total number of elements
    # is not divisible by 4.
    numel = 1024 + 3  # not divisible by 4
    x = torch.randn(numel, device='cuda', dtype=torch.float32)
    min_val = -0.5
    max_val = 0.5

    kernel = build_kernel()
    out_kernel = kernel.forward(x, min_val, max_val)
    out_ref = ref_hardtanh(x, min_val, max_val)
    assert torch.allclose(out_kernel, out_ref, atol=1e-5), "Kernel output differs when numel is non-divisible by vector width."
