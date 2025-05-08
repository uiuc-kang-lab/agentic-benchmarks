
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="test_kernel",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Using vectorized loads/stores with non-float types (e.g. double)
def test_incorrect_vectorization_for_double():
    # Create double-precision input; the kernel’s reinterpret_cast to float4 will be invalid.
    batch_size = 4
    normalized_size = 64  # must be multiple of 4 for vectorized path
    outer_size = batch_size
    # Create tensor of shape (outer_size, normalized_size) in double
    x = torch.randn(outer_size, normalized_size, dtype=torch.double, device='cuda')
    weight = torch.randn(normalized_size, dtype=torch.double, device='cuda')
    bias = torch.randn(normalized_size, dtype=torch.double, device='cuda')
    
    my_kernel = build_kernel()
    # Get kernel result (this is expected to be wrong because of reinterpret_cast issues)
    out_custom = my_kernel.forward(x, weight, bias, 1e-5)
    
    # Use PyTorch's built-in LayerNorm for correct reference result in double precision.
    ln = torch.nn.LayerNorm(normalized_size, eps=1e-5).to(dtype=torch.double, device='cuda')
    # Manually set weight and bias so that we exactly mimic the kernel’s parameters
    with torch.no_grad():
        ln.weight.copy_(weight)
        ln.bias.copy_(bias)
    out_ref = ln(x)
    
    # The bug should cause a significant difference
    assert not torch.allclose(out_custom, out_ref, atol=1e-5), "Kernel unexpectedly produced correct result for double input despite vectorization issue."

# Issue 2: Misaligned memory accesses
def test_misaligned_memory():
    # Create a tensor with extra padding and then slice it to force misalignment.
    # We create a larger tensor and then take a subtensor that is offset by one element.
    batch_size = 4
    normalized_size = 64
    outer_size = batch_size
    # Create a padded tensor for x (padded along last dimension)
    padded = torch.randn(outer_size, normalized_size + 1, dtype=torch.float32, device='cuda')
    # "Misalign" by taking from column 1 onward so that the pointer is offset by 4 bytes.
    x = padded[:, 1:]
    # weight and bias, however, are created normally (and are aligned)
    weight = torch.randn(normalized_size, dtype=torch.float32, device='cuda')
    bias = torch.randn(normalized_size, dtype=torch.float32, device='cuda')
    
    my_kernel = build_kernel()
    out_custom = my_kernel.forward(x, weight, bias, 1e-5)
    
    # Use built-in LayerNorm for reference result.
    ln = torch.nn.LayerNorm(normalized_size, eps=1e-5).to(dtype=torch.float32, device='cuda')
    with torch.no_grad():
        ln.weight.copy_(weight)
        ln.bias.copy_(bias)
    out_ref = ln(x)
    
    # Due to misaligned memory, the kernel should produce a different (wrong) result.
    assert not torch.allclose(out_custom, out_ref, atol=1e-4), "Kernel unexpectedly handled misaligned memory correctly."

# Issue 3: eps precision loss when using double precision
def test_eps_precision_loss():
    batch_size = 4
    normalized_size = 64
    outer_size = batch_size
    # Create double-precision inputs
    x = torch.randn(outer_size, normalized_size, dtype=torch.double, device='cuda')
    weight = torch.randn(normalized_size, dtype=torch.double, device='cuda')
    bias = torch.randn(normalized_size, dtype=torch.double, device='cuda')
    
    # Use a very small eps value in double precision
    eps_double = 1e-7
    my_kernel = build_kernel()
    out_custom = my_kernel.forward(x, weight, bias, eps_double)
    
    # Built-in LayerNorm uses the provided eps as double, so the result should differ if eps was improperly cast.
    ln = torch.nn.LayerNorm(normalized_size, eps=eps_double).to(dtype=torch.double, device='cuda')
    with torch.no_grad():
        ln.weight.copy_(weight)
        ln.bias.copy_(bias)
    out_ref = ln(x)
    
    # If eps is cast to float in the kernel, then for very small eps there will be an observable difference.
    assert not torch.allclose(out_custom, out_ref, atol=1e-5), "Kernel unexpectedly handled eps precision correctly for double input."

# Issue 4: Assumption of contiguous 2D layout; non-contiguous inputs should be handled.
def test_noncontiguous_input():
    batch_size = 4
    normalized_size = 64
    outer_size = batch_size
    # Create a contiguous tensor and then transpose it to break the contiguous layout.
    x = torch.randn(outer_size, normalized_size, dtype=torch.float32, device='cuda')
    # Transpose makes the tensor non-contiguous even though its shape remains the same.
    x_noncontiguous = x.t()  # shape becomes (normalized_size, outer_size)
    # For the test, we want a tensor that still has normalized_size as its inner dimension.
    # So we twist the axes: simulate a higher-D tensor that is non-contiguous.
    x_noncontiguous = x_noncontiguous.t()  # back to original shape but likely non-contiguous
    assert not x_noncontiguous.is_contiguous(), "Tensor is unexpectedly contiguous."

    weight = torch.randn(normalized_size, dtype=torch.float32, device='cuda')
    bias = torch.randn(normalized_size, dtype=torch.float32, device='cuda')
    
    my_kernel = build_kernel()
    out_custom = my_kernel.forward(x_noncontiguous, weight, bias, 1e-5)
    
    ln = torch.nn.LayerNorm(normalized_size, eps=1e-5).to(dtype=torch.float32, device='cuda')
    with torch.no_grad():
        ln.weight.copy_(weight)
        ln.bias.copy_(bias)
    out_ref = ln(x_noncontiguous)
    
    # The kernel will process the data as if it were contiguous in (outer_size, normalized_size)
    # hence the result will differ.
    assert not torch.allclose(out_custom, out_ref, atol=1e-5), "Kernel unexpectedly produced correct result for non-contiguous input."

if __name__ == "__main__":
    pytest.main([__file__])
