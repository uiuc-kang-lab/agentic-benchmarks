
import pytest
import torch
from torch.utils.cpp_extension import load

def build_kernel():
    # Note: Adjust the path to kernel.cu if needed.
    cuda_module = load(
        name="prod_kernel_module",
        sources=["kernel.cu"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: Incorrect base index mapping when reduction dim is not the last dimension.
def test_incorrect_index_mapping():
    # Create a tensor of shape (2, 3, 4). We will reduce along dimension 1 (not the last).
    x = torch.randn(2, 3, 4, device="cuda", dtype=torch.float32)
    # expected output computed by PyTorch: shape becomes (2, 4)
    expected = torch.prod(x, dim=1)
    
    module = build_kernel()
    # our CUDA kernel is invoked with reduction dimension = 1
    output = module.forward(x, 1)
    torch.cuda.synchronize()
    
    # The kernel incorrectly maps indices so the result will be different from expected.
    # This assert is expected to fail if the kernel were used in production.
    assert not torch.allclose(output, expected, atol=1e-5), (
        "Test for incorrect index mapping did not trigger the issue: output unexpectedly matches expected."
    )

# Issue 2: Kernel does not support data types other than float32.
def test_incorrect_dtype_support():
    # Create a tensor of type float64 on CUDA. The kernel always casts to float (32-bit)
    x_double = torch.randn(2, 3, 4, device="cuda", dtype=torch.float64)
    expected = torch.prod(x_double, dim=1)
    
    module = build_kernel()
    # Call kernel with a double tensor. This may silently produce wrong results.
    output = module.forward(x_double, 1)
    torch.cuda.synchronize()
    
    # Force cast expected to float32 to compare with output from kernel;
    # if the kernel were handling types correctly, the result should be close.
    expected_f = expected.to(torch.float32)
    assert not torch.allclose(output, expected_f, atol=1e-5), (
        "Test for dtype checking did not trigger the issue: kernel produced correct results "
        "even with float64 input."
    )

# Issue 3: Lack of validation for a valid reduction dimension.
def test_invalid_reduction_dim():
    x = torch.randn(2, 3, 4, device="cuda", dtype=torch.float32)
    module = build_kernel()
    
    # Pass an invalid dimension (e.g., 5 when x has 3 dimensions). This should trigger an error.
    with pytest.raises(IndexError):
        module.forward(x, 5)
