
import pytest
import torch
from torch.utils.cpp_extension import load
import os

# Build the extension module from kernel.cu
def build_kernel():
    # Assuming kernel.cu is in the same directory as the test file.
    cuda_module = load(
        name="masked_cumsum_cuda",
        sources=[os.path.join(os.path.dirname(__file__), "kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=True,
    )
    return cuda_module

# Issue 1: The kernel does not support half (float16) precision inputs.
def test_half_precision_not_supported():
    my_module = build_kernel()
    # Create half precision input tensor
    batch_size = 8
    L = 100
    x = torch.randn(batch_size, L, device="cuda", dtype=torch.float16)
    mask = torch.randint(0, 2, (batch_size, L), device="cuda").bool()
    with pytest.raises(RuntimeError) as excinfo:
        my_module.forward(x, mask)
    assert "AT_DISPATCH_FLOATING_TYPES" in str(excinfo.value)

# Issue 2: No error-check after kernel launch can lead to silent errors. 
# We simulate a potential launch error by providing a non-contiguous tensor (which is rejected by our TORCH_CHECKs).
def test_non_contiguous_tensor():
    my_module = build_kernel()
    batch_size = 8
    L = 100
    x = torch.randn(batch_size, L, device="cuda")
    mask = torch.randint(0, 2, (batch_size, L), device="cuda").bool()
    # Make x non-contiguous by transposing a 2D tensor
    x_noncontig = x.t()
    mask_noncontig = mask.t()
    with pytest.raises(RuntimeError) as excinfo:
        my_module.forward(x_noncontig, mask_noncontig)
    assert "contiguous" in str(excinfo.value)

# Issue 3 & 4: Testing with a length L that is not a multiple of 4.
# This test checks that the kernel produces correct results even in this unoptimized/unrolled scenario.
def test_non_multiple_of_four_length():
    my_module = build_kernel()
    batch_size = 4
    L = 3  # non multiple of 4
    # Input where cumulative sum is easy to verify.
    x = torch.tensor([[1., 2., 3.],
                      [4., 5., 6.],
                      [7., 8., 9.],
                      [10., 11., 12.]], device="cuda")
    # Create a mask that only selects even-indexed elements (simulate a boolean mask)
    mask = torch.tensor([[True, False, True],
                         [False, True, False],
                         [True, True, False],
                         [False, False, True]], device="cuda")
    # Expected: cumulative sum along last dimension, but only add when mask is True.
    # Row 0: [1, 1, 1+3=4]
    # Row 1: [0, 5, 5]
    # Row 2: [7, 7+8=15, 15]
    # Row 3: [0, 0, 12]
    expected = torch.tensor([[1., 1., 4.],
                             [0., 5., 5.],
                             [7., 15., 15.],
                             [0., 0., 12.]], device="cuda")
    output = my_module.forward(x, mask)
    torch.cuda.synchronize()
    assert torch.allclose(output, expected), f"Output {output} differs from expected {expected}"

# Additional test: Passing an invalid dimension should raise an error.
def test_invalid_dimension():
    my_module = build_kernel()
    batch_size = 8
    L = 100
    x = torch.randn(batch_size, L, device="cuda")
    mask = torch.randint(0, 2, (batch_size, L), device="cuda").bool()
    # Specify an invalid dimension (e.g. 2 for a 2D tensor, valid dims: 0 or 1)
    with pytest.raises(RuntimeError) as excinfo:
        my_module.forward(x, mask, 2)
    assert "Invalid dimension" in str(excinfo.value)
