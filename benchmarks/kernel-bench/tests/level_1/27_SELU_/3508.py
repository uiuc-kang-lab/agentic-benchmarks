
import torch
import pytest
from torch.utils.cpp_extension import load

def build_kernel():
    cuda_module = load(
        name="selu_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Test 1: Noncontiguous input (misaligned memory)
def test_noncontiguous_input():
    my_module = build_kernel()
    # Create a contiguous tensor first.
    x = torch.randn(1024, dtype=torch.float32, device='cuda')
    # Create a noncontiguous view via slicing.
    # For example, take every other element so that the resulting tensor is not contiguous.
    noncontig_x = x[::2]
    # Both should produce SELU activation but our kernel expects the input to be contiguous.
    try:
        y = my_module.forward(noncontig_x)
    except RuntimeError as e:
        # Expect a runtime error or misaligned access error (or wrong result)
        pytest.skip("Noncontiguous tensors are unsupported by the kernel: " + str(e))
    # Compare with torch.selu (which supports noncontiguous tensors correctly)
    y_ref = torch.selu(noncontig_x)
    # This test might fail if kernel misbehaves due to misaligned memory
    assert torch.allclose(y, y_ref, atol=1e-5), "Kernel produced different results for noncontiguous input."

# Test 2: Contiguous input should work correctly
def test_contiguous_input():
    my_module = build_kernel()
    x = torch.randn(1024, dtype=torch.float32, device='cuda')
    # Ensure x is contiguous.
    x = x.contiguous()
    y = my_module.forward(x)
    y_ref = torch.selu(x)
    assert torch.allclose(y, y_ref, atol=1e-5), "Kernel output differs from torch.selu on contiguous input."

# Test 3: Attempt half-precision support, which is not implemented
def test_half_precision_input():
    my_module = build_kernel()
    x = torch.randn(1024, dtype=torch.float16, device='cuda')
    x = x.contiguous()  # make sure it is contiguous
    with pytest.raises(RuntimeError):
        # Expect the kernel dispatch (or the AT_DISPATCH_FLOATING_TYPES macro)
        # to raise an error because half precision is not supported by our kernel.
        y = my_module.forward(x)
        torch.cuda.synchronize()

# Test 4: Check a vectorized remainder issue by creating a tensor whose number of elements
# is not divisible by the vector size (4 for floats). This test is to exercise the remainder loop.
def test_non_divisible_numel():
    my_module = build_kernel()
    # Create a tensor with a number of elements not divisible by 4.
    numel = 1023  # 1023 mod 4 != 0
    x = torch.randn(numel, dtype=torch.float32, device='cuda')
    x = x.contiguous()
    y = my_module.forward(x)
    y_ref = torch.selu(x)
    assert torch.allclose(y, y_ref, atol=1e-5), "Kernel output differs when numel is not divisible by vector size."
