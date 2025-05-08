
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    # This builds the extension from kernel.cu.
    cuda_module = load(
        name="masked_cumsum_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# The function to compare with is torch.cumsum applied after masking.
def ref_masked_cumsum(x, mask, dim):
    return torch.cumsum(x * mask, dim=dim)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_shared_mem_type_with_double():
    # This test addresses Issue 1 by using a non-float32 (double) tensor.
    # If the shared memory allocation type is computed incorrectly, the result may differ from torch.cumsum.
    device = "cuda"
    batch_size, L = 8, 500  # moderate size row
    x = torch.randn(batch_size, L, device=device, dtype=torch.double)
    mask = torch.randint(0, 2, (batch_size, L), device=device, dtype=torch.bool)
    
    # Permute dimensions similar to the kernel host function if needed.
    # For simplicity, we assume cumulative sum along the last dim.
    module = build_kernel()
    # Call the kernel function exposed as 'forward'.
    out_kernel = module.forward(x, mask)
    out_ref = ref_masked_cumsum(x, mask, dim=-1)
    
    # Using a relatively loose tolerance because of possible floating-point ordering differences.
    assert torch.allclose(out_kernel, out_ref, atol=1e-8), (
        f"Double precision output from kernel does not match torch.cumsum.\n"
        f"Max difference: {(out_kernel - out_ref).abs().max().item()}"
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp16_precision_issue():
    # This test addresses Issue 2 by using a float16 tensor.
    # The non-associative nature of the scan may lead to noticeable precision differences relative to torch.cumsum.
    device = "cuda"
    batch_size, L = 8, 1024
    # Using smaller random numbers to reduce the dynamic range challenge.
    x = (torch.randn(batch_size, L, device=device, dtype=torch.float16) * 0.1)
    mask = torch.randint(0, 2, (batch_size, L), device=device, dtype=torch.bool)
    
    module = build_kernel()
    out_kernel = module.forward(x, mask)
    out_ref = ref_masked_cumsum(x, mask, dim=-1)
    
    # For float16, small differences might be expected because of different accumulation order.
    # We set a tolerance that is somewhat loose.
    if not torch.allclose(out_kernel, out_ref, atol=1e-2):
        raise AssertionError(
            f"Float16 output from kernel differs significantly from torch.cumsum."
            f"Max difference: {(out_kernel - out_ref).abs().max().item()}"
        )
